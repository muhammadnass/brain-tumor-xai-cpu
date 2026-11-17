"""
evaluation/database.py
======================
SQLite-backed persistent store for the Phase 2 clinical evaluation study.

Design principles
-----------------
- All data stays local on the Ubuntu laptop; nothing leaves the machine.
- No real patient names, DOBs, or NHS/hospital numbers are ever written to
  the database.  The only patient-side identifier is the case_ref, which
  the system derives from the NIfTI filename (option C) and the evaluator
  can edit before saving.  The caller is responsible for stripping any PHI
  from the filename before passing it here.
- Evaluator credentials are stored as bcrypt hashes; plaintext passwords
  never touch disk.
- All timestamps are UTC ISO-8601.

Schema (v1)
-----------
evaluators
    id            INTEGER PK
    username      TEXT UNIQUE
    password_hash TEXT            (bcrypt, 60 chars)
    display_name  TEXT
    created_at    TEXT

cases
    id            INTEGER PK
    evaluator_id  INTEGER FK → evaluators
    case_ref      TEXT            (derived from T1 filename, PHI-stripped)
    modality_hash TEXT            (SHA-256 of four input files — deduplication key)
    created_at    TEXT
    status        TEXT            ('pending' | 'evaluated' | 'exported')

evaluations
    id                        INTEGER PK
    case_id                   INTEGER FK → cases
    evaluator_id              INTEGER FK → evaluators
    -- Clinical concordance (see forms.py for full question text)
    concordance               TEXT    ('fully_agree'|'partially_agree'|'disagree')
    xai_altered_reading       TEXT    ('reinforced'|'altered'|'no_effect')
    anatomical_plausibility   INTEGER (1–5 Likert)
    tumour_location_correct   INTEGER (0|1 boolean)
    margin_accurate           INTEGER (0|1 boolean)
    free_text_notes           TEXT
    -- SUS (10 items, 1–5 each; scored 0–100 per standard formula)
    sus_1  INTEGER sus_2  INTEGER sus_3  INTEGER sus_4  INTEGER sus_5  INTEGER
    sus_6  INTEGER sus_7  INTEGER sus_8  INTEGER sus_9  INTEGER sus_10 INTEGER
    sus_score REAL          (computed: (sum_odd-5 + 25-sum_even)*2.5)
    -- NASA-TLX (6 dimensions, 0–100 each; raw scores, no weighting)
    nasa_mental_demand     INTEGER
    nasa_physical_demand   INTEGER
    nasa_temporal_demand   INTEGER
    nasa_performance       INTEGER
    nasa_effort            INTEGER
    nasa_frustration       INTEGER
    nasa_raw_score REAL    (mean of six dimensions)
    -- Timestamps
    submitted_at  TEXT
    updated_at    TEXT

disagreements
    id                INTEGER PK
    case_id           INTEGER FK → cases
    evaluator_id      INTEGER FK → evaluators
    region_notes      TEXT    (free text: where the model and surgeon disagreed)
    model_volume_vox  INTEGER
    confidence_score  REAL
    clinician_override TEXT   ('larger'|'smaller'|'different_location'|'absent')
    flag_for_retrain  INTEGER (0|1)
    created_at        TEXT

exports
    id            INTEGER PK
    case_id       INTEGER FK → cases
    pdf_path      TEXT
    nifti_path    TEXT
    exported_at   TEXT
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import bcrypt

from config import RESULTS_DIR

DB_PATH = RESULTS_DIR / "study.db"
_lock = threading.Lock()   # serialise concurrent writes from background threads


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    """Return a thread-local connection with FK enforcement on."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they do not yet exist. Safe to call repeatedly."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluators (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            display_name  TEXT    NOT NULL DEFAULT '',
            created_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cases (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator_id  INTEGER NOT NULL REFERENCES evaluators(id),
            case_ref      TEXT    NOT NULL,
            modality_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'pending',
            UNIQUE(evaluator_id, case_ref)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id                 INTEGER NOT NULL REFERENCES cases(id),
            evaluator_id            INTEGER NOT NULL REFERENCES evaluators(id),
            concordance             TEXT,
            xai_altered_reading     TEXT,
            anatomical_plausibility INTEGER,
            tumour_location_correct INTEGER,
            margin_accurate         INTEGER,
            free_text_notes         TEXT,
            sus_1   INTEGER, sus_2  INTEGER, sus_3  INTEGER,
            sus_4   INTEGER, sus_5  INTEGER, sus_6  INTEGER,
            sus_7   INTEGER, sus_8  INTEGER, sus_9  INTEGER,
            sus_10  INTEGER,
            sus_score               REAL,
            nasa_mental_demand      INTEGER,
            nasa_physical_demand    INTEGER,
            nasa_temporal_demand    INTEGER,
            nasa_performance        INTEGER,
            nasa_effort             INTEGER,
            nasa_frustration        INTEGER,
            nasa_raw_score          REAL,
            submitted_at            TEXT,
            updated_at              TEXT,
            -- surgical outcome comparison (Phase 2 clinical utility paper)
            surgical_extent         TEXT,
            margin_agreement        TEXT,
            risk_category_match     TEXT,
            intraop_notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS disagreements (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id           INTEGER NOT NULL REFERENCES cases(id),
            evaluator_id      INTEGER NOT NULL REFERENCES evaluators(id),
            region_notes      TEXT,
            model_volume_vox  INTEGER,
            confidence_score  REAL,
            clinician_override TEXT,
            flag_for_retrain  INTEGER NOT NULL DEFAULT 0,
            created_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS exports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id      INTEGER NOT NULL REFERENCES cases(id),
            pdf_path     TEXT,
            nifti_path   TEXT,
            exported_at  TEXT    NOT NULL
        );
        """)

    # -----------------------------------------------------------------------
    # Incremental migrations: add columns that may not exist in older databases.
    # ALTER TABLE ADD COLUMN is idempotent-safe: we catch the OperationalError
    # that SQLite raises when a column already exists.
    # -----------------------------------------------------------------------
    _add_column_if_missing(conn, "evaluations", "surgical_extent",    "TEXT")
    _add_column_if_missing(conn, "evaluations", "margin_agreement",   "TEXT")
    _add_column_if_missing(conn, "evaluations", "risk_category_match","TEXT")
    _add_column_if_missing(conn, "evaluations", "intraop_notes",      "TEXT")


def _add_column_if_missing(conn, table: str, column: str, coltype: str):
    """Add a column to *table* if it does not already exist (migration helper)."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    except Exception:
        pass  # column already exists or table doesn't exist yet — both are fine


# ===========================================================================
# EVALUATOR AUTH
# ===========================================================================

def create_evaluator(username: str, password: str,
                     display_name: str = "") -> int:
    """
    Create a new evaluator account.  Returns the new row id.
    Raises ValueError if the username already exists.
    """
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with _lock, get_connection() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO evaluators (username, password_hash, display_name, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, pw_hash, display_name or username, _utcnow())
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")


def verify_evaluator(username: str, password: str) -> dict | None:
    """
    Check credentials.  Returns evaluator row as dict, or None on failure.
    Timing-safe: bcrypt.checkpw runs regardless of whether the user exists.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM evaluators WHERE username = ?", (username,)
        ).fetchone()

    # Always run checkpw to prevent timing oracle even when user not found
    dummy_hash = b"$2b$12$invalidhashpaddingtomatch60chars00000000000000000000000000"
    stored = row["password_hash"].encode() if row else dummy_hash
    pw_bytes = password.encode()
    valid = bcrypt.checkpw(pw_bytes, stored)

    if row and valid:
        return dict(row)
    return None


def list_evaluators() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, created_at FROM evaluators"
        ).fetchall()
    return [dict(r) for r in rows]


def change_password(evaluator_id: int, new_password: str):
    pw_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    with _lock, get_connection() as conn:
        conn.execute(
            "UPDATE evaluators SET password_hash = ? WHERE id = ?",
            (pw_hash, evaluator_id)
        )


# ===========================================================================
# CASES
# ===========================================================================

def get_or_create_case(evaluator_id: int, case_ref: str,
                       modality_hash: str) -> dict:
    """
    Return existing case for (evaluator_id, case_ref) if it exists,
    otherwise create it.  This is the longitudinal continuity mechanism:
    the same evaluator returning to the same case_ref resumes the prior record.
    """
    with _lock, get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE evaluator_id = ? AND case_ref = ?",
            (evaluator_id, case_ref)
        ).fetchone()
        if row:
            return dict(row)
        cur = conn.execute(
            "INSERT INTO cases (evaluator_id, case_ref, modality_hash, created_at, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (evaluator_id, case_ref, modality_hash, _utcnow())
        )
        row = conn.execute(
            "SELECT * FROM cases WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)


def list_cases(evaluator_id: int) -> list[dict]:
    """Return all cases for an evaluator, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT c.*, "
            "  (SELECT COUNT(*) FROM evaluations e WHERE e.case_id = c.id) AS eval_count "
            "FROM cases c WHERE c.evaluator_id = ? ORDER BY c.created_at DESC",
            (evaluator_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_case_status(case_id: int, status: str):
    assert status in ('pending', 'evaluated', 'exported')
    with _lock, get_connection() as conn:
        conn.execute("UPDATE cases SET status = ? WHERE id = ?",
                     (status, case_id))


# ===========================================================================
# EVALUATIONS
# ===========================================================================

def _compute_sus(items: dict) -> float:
    """
    Standard SUS scoring formula.
    Odd items (1,3,5,7,9): subtract 1 from the raw score.
    Even items (2,4,6,8,10): subtract the raw score from 5.
    Sum all 10 adjusted scores and multiply by 2.5 → range 0–100.
    """
    odd  = sum(items.get(f"sus_{i}", 3) - 1 for i in (1, 3, 5, 7, 9))
    even = sum(5 - items.get(f"sus_{i}", 3) for i in (2, 4, 6, 8, 10))
    return (odd + even) * 2.5


def _compute_nasa(items: dict) -> float:
    dims = ["nasa_mental_demand", "nasa_physical_demand", "nasa_temporal_demand",
            "nasa_performance", "nasa_effort", "nasa_frustration"]
    vals = [items.get(d, 50) for d in dims]
    return sum(vals) / len(vals)


def save_evaluation(case_id: int, evaluator_id: int, data: dict) -> int:
    """
    Insert or replace a complete evaluation record.
    data keys map directly to column names.
    SUS and NASA-TLX composite scores are computed here.
    Returns the evaluation row id.
    """
    sus_score   = _compute_sus(data)
    nasa_score  = _compute_nasa(data)
    now         = _utcnow()

    with _lock, get_connection() as conn:
        # Upsert: if a previous draft exists for this case, overwrite it
        existing = conn.execute(
            "SELECT id FROM evaluations WHERE case_id = ? AND evaluator_id = ?",
            (case_id, evaluator_id)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE evaluations SET
                    concordance=:concordance,
                    xai_altered_reading=:xai_altered_reading,
                    anatomical_plausibility=:anatomical_plausibility,
                    tumour_location_correct=:tumour_location_correct,
                    margin_accurate=:margin_accurate,
                    free_text_notes=:free_text_notes,
                    sus_1=:sus_1,   sus_2=:sus_2,   sus_3=:sus_3,
                    sus_4=:sus_4,   sus_5=:sus_5,   sus_6=:sus_6,
                    sus_7=:sus_7,   sus_8=:sus_8,   sus_9=:sus_9,
                    sus_10=:sus_10, sus_score=:sus_score,
                    nasa_mental_demand=:nasa_mental_demand,
                    nasa_physical_demand=:nasa_physical_demand,
                    nasa_temporal_demand=:nasa_temporal_demand,
                    nasa_performance=:nasa_performance,
                    nasa_effort=:nasa_effort,
                    nasa_frustration=:nasa_frustration,
                    nasa_raw_score=:nasa_raw_score,
                    surgical_extent=:surgical_extent,
                    margin_agreement=:margin_agreement,
                    risk_category_match=:risk_category_match,
                    intraop_notes=:intraop_notes,
                    updated_at=:updated_at
                WHERE id=:id
            """, {**data, 'sus_score': sus_score, 'nasa_raw_score': nasa_score,
                  'updated_at': now, 'id': existing['id'],
                  'surgical_extent':    data.get('surgical_extent', 'not_available'),
                  'margin_agreement':   data.get('margin_agreement', 'not_assessed'),
                  'risk_category_match':data.get('risk_category_match', 'not_assessed'),
                  'intraop_notes':      data.get('intraop_notes', ''),
                  })
            row_id = existing['id']
        else:
            cur = conn.execute("""
                INSERT INTO evaluations (
                    case_id, evaluator_id,
                    concordance, xai_altered_reading,
                    anatomical_plausibility, tumour_location_correct,
                    margin_accurate, free_text_notes,
                    sus_1, sus_2, sus_3, sus_4, sus_5,
                    sus_6, sus_7, sus_8, sus_9, sus_10, sus_score,
                    nasa_mental_demand, nasa_physical_demand,
                    nasa_temporal_demand, nasa_performance,
                    nasa_effort, nasa_frustration, nasa_raw_score,
                    surgical_extent, margin_agreement, risk_category_match,
                    intraop_notes, submitted_at, updated_at
                ) VALUES (
                    :case_id, :evaluator_id,
                    :concordance, :xai_altered_reading,
                    :anatomical_plausibility, :tumour_location_correct,
                    :margin_accurate, :free_text_notes,
                    :sus_1, :sus_2, :sus_3, :sus_4, :sus_5,
                    :sus_6, :sus_7, :sus_8, :sus_9, :sus_10, :sus_score,
                    :nasa_mental_demand, :nasa_physical_demand,
                    :nasa_temporal_demand, :nasa_performance,
                    :nasa_effort, :nasa_frustration, :nasa_raw_score,
                    :surgical_extent, :margin_agreement, :risk_category_match,
                    :intraop_notes, :submitted_at, :updated_at
                )
            """, {**data, 'case_id': case_id, 'evaluator_id': evaluator_id,
                  'sus_score': sus_score, 'nasa_raw_score': nasa_score,
                  'submitted_at': now, 'updated_at': now,
                  'surgical_extent':    data.get('surgical_extent', 'not_available'),
                  'margin_agreement':   data.get('margin_agreement', 'not_assessed'),
                  'risk_category_match':data.get('risk_category_match', 'not_assessed'),
                  'intraop_notes':      data.get('intraop_notes', ''),
                  })
            row_id = cur.lastrowid

        update_case_status(case_id, 'evaluated')
        return row_id


def get_evaluation(case_id: int, evaluator_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE case_id = ? AND evaluator_id = ?",
            (case_id, evaluator_id)
        ).fetchone()
    return dict(row) if row else None


# ===========================================================================
# DISAGREEMENTS  (Phase 2 item 3 — model improvement loop)
# ===========================================================================

def log_disagreement(case_id: int, evaluator_id: int, data: dict) -> int:
    """
    Record a case where the model and clinician disagreed.
    data keys: region_notes, model_volume_vox, confidence_score,
               clinician_override, flag_for_retrain (0|1)
    """
    with _lock, get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO disagreements
                (case_id, evaluator_id, region_notes, model_volume_vox,
                 confidence_score, clinician_override, flag_for_retrain, created_at)
            VALUES
                (:case_id, :evaluator_id, :region_notes, :model_volume_vox,
                 :confidence_score, :clinician_override, :flag_for_retrain, :created_at)
        """, {**data, 'case_id': case_id, 'evaluator_id': evaluator_id,
              'created_at': _utcnow()})
        return cur.lastrowid


def get_flagged_for_retrain() -> list[dict]:
    """Return all disagreements flagged for model retraining (item 3)."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT d.*, c.case_ref, e.username
            FROM disagreements d
            JOIN cases c ON c.id = d.case_id
            JOIN evaluators e ON e.id = d.evaluator_id
            WHERE d.flag_for_retrain = 1
            ORDER BY d.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# EXPORTS
# ===========================================================================

def record_export(case_id: int, pdf_path: str, nifti_path: str) -> int:
    with _lock, get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO exports (case_id, pdf_path, nifti_path, exported_at) "
            "VALUES (?, ?, ?, ?)",
            (case_id, pdf_path, nifti_path, _utcnow())
        )
        update_case_status(case_id, 'exported')
        return cur.lastrowid


def list_exports(case_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM exports WHERE case_id = ? ORDER BY exported_at DESC",
            (case_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# AGGREGATE (for admin portal)
# ===========================================================================

def get_study_summary() -> dict:
    """High-level summary for the hospital MD admin view."""
    with get_connection() as conn:
        n_evaluators = conn.execute(
            "SELECT COUNT(*) FROM evaluators"
        ).fetchone()[0]
        n_cases = conn.execute(
            "SELECT COUNT(*) FROM cases"
        ).fetchone()[0]
        n_evaluations = conn.execute(
            "SELECT COUNT(*) FROM evaluations"
        ).fetchone()[0]
        avg_sus = conn.execute(
            "SELECT AVG(sus_score) FROM evaluations WHERE sus_score IS NOT NULL"
        ).fetchone()[0]
        avg_nasa = conn.execute(
            "SELECT AVG(nasa_raw_score) FROM evaluations WHERE nasa_raw_score IS NOT NULL"
        ).fetchone()[0]
        concordance_counts = conn.execute("""
            SELECT concordance, COUNT(*) as n
            FROM evaluations
            WHERE concordance IS NOT NULL
            GROUP BY concordance
        """).fetchall()
        flagged = conn.execute(
            "SELECT COUNT(*) FROM disagreements WHERE flag_for_retrain = 1"
        ).fetchone()[0]

    return {
        'n_evaluators':    n_evaluators,
        'n_cases':         n_cases,
        'n_evaluations':   n_evaluations,
        'avg_sus_score':   round(avg_sus, 1) if avg_sus else None,
        'avg_nasa_score':  round(avg_nasa, 1) if avg_nasa else None,
        'concordance':     {r['concordance']: r['n'] for r in concordance_counts},
        'flagged_retrain': flagged,
    }


def export_all_deidentified() -> list[dict]:
    """
    Full evaluation dataset export — de-identified.
    Evaluator username replaced with a stable anonymous code (E01, E02, …).
    Case ref is already PHI-free by design (derived from filename, never
    from hospital record or patient name).
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                ev.id           AS evaluation_id,
                'E' || printf('%02d', ev.evaluator_id) AS evaluator_code,
                c.case_ref,
                c.created_at    AS case_created_at,
                c.status        AS case_status,
                ev.concordance,
                ev.xai_altered_reading,
                ev.anatomical_plausibility,
                ev.tumour_location_correct,
                ev.margin_accurate,
                ev.sus_1, ev.sus_2, ev.sus_3, ev.sus_4, ev.sus_5,
                ev.sus_6, ev.sus_7, ev.sus_8, ev.sus_9, ev.sus_10,
                ev.sus_score,
                ev.nasa_mental_demand,
                ev.nasa_physical_demand,
                ev.nasa_temporal_demand,
                ev.nasa_performance,
                ev.nasa_effort,
                ev.nasa_frustration,
                ev.nasa_raw_score,
                ev.surgical_extent,
                ev.margin_agreement,
                ev.risk_category_match,
                ev.intraop_notes,
                ev.submitted_at,
                ev.free_text_notes
            FROM evaluations ev
            JOIN cases c ON c.id = ev.case_id
            ORDER BY ev.submitted_at
        """).fetchall()
    return [dict(r) for r in rows]


def export_evaluator_session(evaluator_id: int) -> list[dict]:
    """
    Per-evaluator session export — all cases evaluated by this evaluator.
    Used for the 'Download session CSV' button in the Streamlit UI.
    De-identified: no patient name, no evaluator username.
    Includes surgical outcome comparison fields for the clinical utility paper.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                ev.id                                    AS evaluation_id,
                'E' || printf('%02d', ev.evaluator_id)   AS evaluator_code,
                c.case_ref,
                c.created_at                             AS case_created_at,
                c.status                                 AS case_status,
                ev.concordance,
                ev.xai_altered_reading,
                ev.anatomical_plausibility,
                ev.tumour_location_correct,
                ev.margin_accurate,
                ev.sus_score,
                ev.nasa_raw_score,
                ev.surgical_extent,
                ev.margin_agreement,
                ev.risk_category_match,
                ev.intraop_notes,
                ev.submitted_at,
                ev.free_text_notes
            FROM evaluations ev
            JOIN cases c ON c.id = ev.case_id
            WHERE ev.evaluator_id = ?
            ORDER BY ev.submitted_at
        """, (evaluator_id,)).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# ADMIN SEEDING
# ===========================================================================

ADMIN_SEED = {
    "username":     "sunusinasir@gmail.com",
    "password":     "%Password1",
    "display_name": "Study Administrator",
}


def seed_admin():
    """
    Create the admin account if it does not exist.
    Called once at app startup. Safe to call repeatedly.
    """
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT id FROM evaluators WHERE username = ?",
            (ADMIN_SEED["username"],)
        ).fetchone()
    if not exists:
        try:
            create_evaluator(
                username=ADMIN_SEED["username"],
                password=ADMIN_SEED["password"],
                display_name=ADMIN_SEED["display_name"],
            )
            # Mark as admin in a separate metadata table
            with _lock, get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS admin_flags (
                        evaluator_id INTEGER PRIMARY KEY
                            REFERENCES evaluators(id)
                    )
                """)
                conn.execute(
                    "INSERT OR IGNORE INTO admin_flags (evaluator_id) "
                    "SELECT id FROM evaluators WHERE username = ?",
                    (ADMIN_SEED["username"],)
                )
            import logging
            logging.getLogger(__name__).info(
                f"Admin account seeded: {ADMIN_SEED['username']}"
            )
        except ValueError:
            pass  # Already exists from a prior run


def is_admin(evaluator_id: int) -> bool:
    """Return True if this evaluator has admin privileges."""
    with get_connection() as conn:
        # Ensure admin_flags table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_flags (
                evaluator_id INTEGER PRIMARY KEY
                    REFERENCES evaluators(id)
            )
        """)
        row = conn.execute(
            "SELECT 1 FROM admin_flags WHERE evaluator_id = ?",
            (evaluator_id,)
        ).fetchone()
    return row is not None


# Initialise tables and seed admin when this module is imported
init_db()
seed_admin()
