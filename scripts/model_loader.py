#!/usr/bin/env python3
"""
model_loader.py -- single place where a trained checkpoint is turned into a
model. Imported by measure_resources.py and evaluate_cross_population.py.

Keep this file alongside those scripts (both resolve it via sys.path[0],
which Python sets to the script's own directory). It exists so that the
loading logic is defined once: a measurement taken against the wrong
architecture is worse than no measurement, and having two copies of this
logic is how that happens.

Design rules, deliberate:

  * No fallback architecture. If the project's UNet3D cannot be imported,
    the script exits. It does not substitute a stand-in network.
  * Layer widths are read from the checkpoint, never hard-coded, so the
    model built here is the model that was trained.
  * Any missing or unexpected key from load_state_dict is fatal. A
    partially loaded network still runs and still produces plausible
    numbers, which is what makes it dangerous.
"""

import os
import sys

import torch


def load_trained_model(checkpoint_path, project_root=None, features=None,
                       provenance=None, expect_threshold=None):
    """Build the project's UNet3D and load a trained checkpoint into it.

    Returns (model, meta). Exits the process on any inconsistency.
    """
    prov = provenance if provenance is not None else {}

    root = os.path.abspath(project_root or os.getcwd())
    for path in (os.path.join(root, "src"), root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    prov["project_root"] = root

    try:
        from ai_models.segmentation import UNet3D
    except ImportError as exc:
        sys.exit(
            "Cannot import UNet3D from ai_models.segmentation (%s).\n"
            "Looked under %s and %s.\n"
            "Run from the project root, or pass --project-root. This script "
            "will not substitute a stand-in architecture."
            % (exc, os.path.join(root, "src"), root))
    prov["model_source"] = "ai_models.segmentation.UNet3D"

    if not os.path.exists(checkpoint_path):
        sys.exit("checkpoint not found: %s" % checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state, dict) or "model_state_dict" not in state:
        sys.exit(
            "checkpoint %s has no 'model_state_dict' key (found: %s). "
            "Refusing to guess." % (checkpoint_path,
                                    sorted(state)[:10]
                                    if isinstance(state, dict) else type(state)))

    feats = features if features is not None else state.get("features")
    if feats is None:
        sys.exit(
            "checkpoint %s does not record a 'features' tuple and none was "
            "supplied. Pass --features rather than allowing a default to be "
            "assumed." % checkpoint_path)
    feats = tuple(int(f) for f in feats)

    model = UNet3D(features=feats)
    try:
        missing, unexpected = model.load_state_dict(
            state["model_state_dict"], strict=False)
    except RuntimeError as exc:
        # strict=False tolerates absent keys but not shape mismatches, which
        # mean the widths are wrong for this checkpoint.
        sys.exit(
            "ABORT: tensor shapes in %s do not match UNet3D(features=%s).\n"
            "%s\n"
            "The layer widths are wrong for this checkpoint. If --features "
            "was passed, drop it and let the widths be read from the "
            "checkpoint." % (checkpoint_path, feats, exc))
    if missing or unexpected:
        sys.exit(
            "ABORT: checkpoint %s did not load cleanly into "
            "UNet3D(features=%s): %d missing key(s), %d unexpected key(s).\n"
            "  missing:    %s\n"
            "  unexpected: %s\n"
            "The architecture does not match the checkpoint. Any measurement "
            "taken now would describe the wrong model."
            % (checkpoint_path, feats, len(missing), len(unexpected),
               list(missing)[:6], list(unexpected)[:6]))

    model.eval()

    meta = {
        "checkpoint": checkpoint_path,
        "features": list(feats),
        "fold": state.get("fold"),
        "epoch": state.get("epoch"),
        "val_dice": state.get("val_dice"),
        "threshold": state.get("threshold"),
        "parameters_total": sum(p.numel() for p in model.parameters()),
    }
    prov["checkpoint_meta"] = meta

    if expect_threshold is not None and meta["threshold"] is not None:
        if abs(float(meta["threshold"]) - float(expect_threshold)) > 1e-9:
            sys.exit(
                "ABORT: checkpoint records threshold %s but this script uses "
                "%s. Reported figures must be computed at the threshold the "
                "model was evaluated under."
                % (meta["threshold"], expect_threshold))

    print("loaded %s | features=%s | %s params | fold=%s epoch=%s val_dice=%s"
          % (os.path.basename(checkpoint_path), feats,
             format(meta["parameters_total"], ","), meta["fold"],
             meta["epoch"], meta["val_dice"]), file=sys.stderr)
    return model, meta


def resolve_layer(model, name):
    """Resolve an exact module name. Exact match only: prefix matching hooks
    every matching layer and is what caused the memory exhaustion described
    in Section 3.16."""
    table = dict(model.named_modules())
    if name in table:
        return table[name]
    hint = [k for k in table if name in k][:12]
    sys.exit(
        "layer %r not found (exact match required). Candidates containing "
        "that string:\n  %s\nRun with --list-layers for all module names."
        % (name, "\n  ".join(hint) if hint else "(none)"))


# ------------------------------------------------------------------ #
# Fold-split reading
# ------------------------------------------------------------------ #
def _fold_entries(cfg, folds_path):
    """Return the list of fold entries, whatever wrapper they arrived in."""
    if isinstance(cfg, dict) and isinstance(cfg.get("folds"), list):
        return cfg["folds"], "cfg['folds'] list"
    if isinstance(cfg, list):
        return cfg, "top-level list"
    sys.exit(
        "Unrecognised fold-file structure in %s (top-level keys: %s). "
        "Expected {'folds': [{'fold': 0, 'train': [...], 'val': [...]}, ...]}. "
        "Refusing to guess." % (folds_path,
                                sorted(cfg)[:10] if isinstance(cfg, dict)
                                else type(cfg).__name__))


def _ids(entry, key_options, fold_index, folds_path):
    for key in key_options:
        if key in entry:
            value = entry[key]
            if not isinstance(value, list):
                sys.exit("fold %d in %s: '%s' is not a list"
                         % (fold_index, folds_path, key))
            return value
    sys.exit("fold %d in %s has none of the keys %s"
             % (fold_index, folds_path, key_options))


def load_fold_split(folds_path, fold_index, provenance=None):
    """Read one fold from the cross-validation split file.

    Expects the structure written by train_cv.py:

        {"seed": 42, "n_folds": 3, "n_cases": 484,
         "folds": [{"fold": 0, "train": [...], "val": [...]}, ...]}

    Fold entries live inside the "folds" list and are addressed by list
    position, not by a top-level key. The entry's own "fold" field is
    checked against the requested index, so a reordered file is caught
    rather than silently evaluating the wrong split.

    Returns (train_ids, val_ids, meta).
    """
    import json

    with open(folds_path) as fh:
        cfg = json.load(fh)
    entries, shape = _fold_entries(cfg, folds_path)

    if not 0 <= fold_index < len(entries):
        sys.exit("fold %d requested but %s contains %d fold(s)"
                 % (fold_index, folds_path, len(entries)))
    entry = entries[fold_index]

    declared = entry.get("fold")
    if declared is not None and int(declared) != int(fold_index):
        sys.exit(
            "ABORT: entry at position %d of %s declares itself fold %s. The "
            "file order does not match the fold numbering, so evaluating by "
            "position would use the wrong split."
            % (fold_index, folds_path, declared))

    train = _ids(entry, ("train", "training"), fold_index, folds_path)
    val = _ids(entry, ("val", "validation", "valid"), fold_index, folds_path)

    overlap = set(train) & set(val)
    if overlap:
        sys.exit(
            "ABORT: fold %d of %s has %d case(s) in both train and val (e.g. "
            "%s). The manuscript states the splits are free of leakage."
            % (fold_index, folds_path, len(overlap), sorted(overlap)[:5]))

    meta = {"folds_file": folds_path, "structure": shape, "fold": fold_index,
            "n_train": len(train), "n_val": len(val),
            "seed": cfg.get("seed") if isinstance(cfg, dict) else None,
            "n_folds": cfg.get("n_folds") if isinstance(cfg, dict) else len(entries),
            "n_cases": cfg.get("n_cases") if isinstance(cfg, dict) else None}
    if provenance is not None:
        provenance.setdefault("fold_splits", {})[str(fold_index)] = meta
    return train, val, meta


def verify_fold_partition(folds_path):
    """Check that the validation splits partition the cohort: every case in
    exactly one val set, none in two. This is the claim made in Section 4.1,
    so it is worth checking rather than assuming."""
    import json

    with open(folds_path) as fh:
        cfg = json.load(fh)
    entries, _ = _fold_entries(cfg, folds_path)

    seen, duplicated = set(), set()
    for i in range(len(entries)):
        _, val, _ = load_fold_split(folds_path, i)
        duplicated |= seen & set(val)
        seen |= set(val)

    if duplicated:
        sys.exit(
            "ABORT: %d case(s) appear in more than one validation split of %s "
            "(e.g. %s). Cases would be counted twice in the pooled aggregate."
            % (len(duplicated), folds_path, sorted(duplicated)[:5]))

    declared = cfg.get("n_cases") if isinstance(cfg, dict) else None
    if declared is not None and len(seen) != int(declared):
        sys.exit(
            "ABORT: validation splits of %s cover %d unique cases but the "
            "file declares n_cases = %s. Not every case would be evaluated "
            "exactly once." % (folds_path, len(seen), declared))

    print("fold partition verified: %d folds, %d unique validation cases, "
          "no case in two folds" % (len(entries), len(seen)), file=sys.stderr)
    return len(seen)


# ------------------------------------------------------------------ #
# Project explainer loading
# ------------------------------------------------------------------ #
def _add_paths(project_root):
    root = os.path.abspath(project_root or os.getcwd())
    for path in (os.path.join(root, "src"), root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    return root


def _hook_census(model):
    """Which modules currently carry forward hooks, and how many."""
    return {name: len(m._forward_hooks)
            for name, m in model.named_modules() if m._forward_hooks}


def _declared_layers(obj):
    """Any attribute that looks like a record of the hooked layer names."""
    for attr in ("_registered_layers", "registered_layers", "target_layers",
                 "layer_names", "layers"):
        val = getattr(obj, attr, None)
        if isinstance(val, (list, tuple)) and all(isinstance(v, str)
                                                  for v in val):
            return attr, list(val)
        if isinstance(val, str):
            return attr, [val]
    return None, None


def release_project_gradcam(model, obj):
    """Remove the forward hooks a GradCAM3D registered at construction.

    Deleting the object is NOT sufficient: the module holds the hook in
    its _forward_hooks dict, so a deleted explainer keeps firing. Left
    unreleased across a loop over layers this accumulates hooks on the
    model, which is the memory-exhaustion mechanism described in Section
    3.16, and makes later no-grad forward passes fail inside a stale hook.
    """
    for closer in ("remove", "close", "remove_hooks", "release"):
        fn = getattr(obj, closer, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    added = getattr(obj, "_loader_added_hooks", None) or {}
    table = dict(model.named_modules())
    for name, ids in added.items():
        module = table.get(name)
        if module is None:
            continue
        for hid in ids:
            module._forward_hooks.pop(hid, None)
    for h in getattr(obj, "_handles", []) or []:
        try:
            h.remove()
        except Exception:
            pass


def load_project_gradcam(model, layer_name, project_root=None,
                         provenance=None):
    """Construct the project's own GradCAM3D for exactly one layer.

    The layer is passed as a SINGLE-ELEMENT LIST, not a string. GradCAM3D
    selects modules with `if name in self.target_layers`; given a string
    that is substring containment, not equality, so a request for
    "decoder.0" also matches the container "decoder" AND the root module,
    whose name is the empty string and which is a substring of every name.
    The root's forward hook fires last, so the captured activation is the
    model's own output whatever layer was asked for, and every layer
    yields an identical map.

    After construction the hooks are audited: exactly the requested module
    must carry one, and nothing else may. Anything else aborts.
    """
    root = _add_paths(project_root)
    try:
        from xai_explainers.attention_visualizer import GradCAM3D
    except ImportError as exc:
        sys.exit(
            "Cannot import GradCAM3D from xai_explainers.attention_visualizer "
            "(%s).\nLooked under %s and %s.\nThis script will not substitute "
            "its own Grad-CAM: the paper documents yours."
            % (exc, os.path.join(root, "src"), root))

    if layer_name not in dict(model.named_modules()):
        sys.exit("layer %r is not a module of this model (exact match "
                 "required)" % layer_name)

    before_ids = {name: set(m._forward_hooks)
                  for name, m in model.named_modules()}
    before = _hook_census(model)
    errors, obj, used = [], None, None
    # List forms first. String forms are last resort and are audited below.
    for label, build in (
            ("GradCAM3D(model, [layer])",
             lambda: GradCAM3D(model, [layer_name])),
            ("GradCAM3D(model, target_layers=[layer])",
             lambda: GradCAM3D(model, target_layers=[layer_name])),
            ("GradCAM3D(model, target_layer=[layer])",
             lambda: GradCAM3D(model, target_layer=[layer_name])),
            ("GradCAM3D(model, layer)",
             lambda: GradCAM3D(model, layer_name)),
            ("GradCAM3D(model, target_layer=layer)",
             lambda: GradCAM3D(model, target_layer=layer_name))):
        try:
            obj = build()
            used = label
            break
        except Exception as exc:
            errors.append("%s -> %r" % (label, exc))
    if obj is None:
        sys.exit("Could not construct GradCAM3D. Attempts:\n  %s"
                 % "\n  ".join(errors))

    if not hasattr(obj, "generate_cam"):
        sys.exit("GradCAM3D has no generate_cam(); available: %s"
                 % [a for a in dir(obj) if not a.startswith("_")])

    # --- audit ---
    attr, declared = _declared_layers(obj)
    if declared is not None and declared != [layer_name]:
        sys.exit(
            "ABORT: GradCAM3D.%s is %r after requesting %r. The explainer is "
            "not targeting the layer that was asked for, so any attribution "
            "it produces would describe a different part of the network."
            % (attr, declared, layer_name))

    after = _hook_census(model)
    added = {k: after[k] - before.get(k, 0) for k in after
             if after[k] - before.get(k, 0) > 0}
    if added:
        if set(added) != {layer_name} or added[layer_name] != 1:
            sys.exit(
                "ABORT: constructing GradCAM3D for %r registered forward "
                "hooks on %s.\nExactly one hook on the requested module was "
                "expected. A hook on the root module (name '') captures the "
                "model output rather than the layer, and every layer then "
                "yields an identical map. Construct with a single-element "
                "list, not a string."
                % (layer_name,
                   {(k or "<root module>"): v for k, v in added.items()}))
        verified = "hook audit: 1 forward hook on %r, none elsewhere" % layer_name
    elif declared is not None:
        verified = ("no hooks registered at construction (registered per "
                    "call); %s == [%r]" % (attr, layer_name))
    else:
        verified = None
        print("WARNING: GradCAM3D registered no hooks at construction and "
              "exposes no attribute recording its target layer, so the "
              "targeting of %r could not be verified. Confirm by hand that "
              "different layers give different maps before reporting "
              "per-layer results." % layer_name, file=sys.stderr)

    # Record the hooks this construction added so they can be released.
    obj._loader_added_hooks = {
        name: set(m._forward_hooks) - before_ids.get(name, set())
        for name, m in model.named_modules()
        if set(m._forward_hooks) - before_ids.get(name, set())}

    if provenance is not None:
        provenance["gradcam_implementation"] = (
            "xai_explainers.attention_visualizer.GradCAM3D via %s, "
            "called as generate_cam(x, target_class)" % used)
        provenance.setdefault("gradcam_targeting", {})[layer_name] = verified

    print("Grad-CAM: %s via %s%s"
          % (layer_name, used, ("  [%s]" % verified) if verified else ""),
          file=sys.stderr)
    return obj


def load_project_explainers(model, layer_name, ig_steps=20,
                            project_root=None, provenance=None):
    """Construct all three attribution methods from the project's own code.

    Confirmed signatures:

        xai_explainers.attention_visualizer:GradCAM3D(model, layer)
            .generate_cam(input_tensor, target_class) -> [D,H,W] ndarray
        xai_explainers.gradient_methods:VanillaGradient(model)
            .generate(input_tensor, target_class=1) -> [D,H,W] ndarray
        xai_explainers.gradient_methods:IntegratedGradients(model, steps=20)
            .generate(input_tensor, target_class=1) -> [D,H,W] ndarray

    Note that IntegratedGradients takes its step count at construction, not
    at call time, so the timed call covers all steps without receiving the
    count.

    Every method reported in the paper comes from here. There is no
    fallback: timing a reimplementation would report latencies for code
    that is not the code the manuscript documents.

    Returns {name: callable(x, target_class) -> ndarray}.
    """
    root = _add_paths(project_root)
    cam = load_project_gradcam(model, layer_name, project_root=root,
                               provenance=provenance)

    try:
        from xai_explainers.gradient_methods import (VanillaGradient,
                                                     IntegratedGradients)
    except ImportError as exc:
        sys.exit(
            "Cannot import VanillaGradient / IntegratedGradients from "
            "xai_explainers.gradient_methods (%s).\nLooked under %s and %s.\n"
            "This script will not substitute its own implementations."
            % (exc, os.path.join(root, "src"), root))

    try:
        vg = VanillaGradient(model)
    except Exception as exc:
        sys.exit("Could not construct VanillaGradient(model): %r" % (exc,))
    try:
        ig = IntegratedGradients(model, steps=ig_steps)
    except Exception as exc:
        sys.exit("Could not construct IntegratedGradients(model, steps=%d): %r"
                 % (ig_steps, exc))

    for name, obj, attr in (("GradCAM3D", cam, "generate_cam"),
                            ("VanillaGradient", vg, "generate"),
                            ("IntegratedGradients", ig, "generate")):
        if not hasattr(obj, attr):
            sys.exit("%s has no %s(); available: %s"
                     % (name, attr, [a for a in dir(obj)
                                     if not a.startswith("_")]))

    impl = {
        "grad_cam": ("xai_explainers.attention_visualizer:GradCAM3D"
                     ".generate_cam(x, target_class)"),
        "vanilla_gradient": ("xai_explainers.gradient_methods:VanillaGradient"
                             ".generate(x, target_class)"),
        "integrated_gradients": (
            "xai_explainers.gradient_methods:IntegratedGradients(steps=%d)"
            ".generate(x, target_class)" % ig_steps),
    }
    if provenance is not None:
        provenance["implementations"] = impl
        provenance["ig_steps_at_construction"] = ig_steps

    print("explainers constructed from project code:", file=sys.stderr)
    for k, v in impl.items():
        print("  %-22s %s" % (k, v), file=sys.stderr)

    return ({"grad_cam": lambda x, c: cam.generate_cam(x, c),
             "vanilla_gradient": lambda x, c: vg.generate(x, c),
             "integrated_gradients": lambda x, c: ig.generate(x, c)},
            impl)


def load_named_model(spec, kwargs, checkpoint, project_root=None,
                     provenance=None):
    """Construct an arbitrary architecture from 'module.path:ClassName' and
    load a checkpoint into it.

    Used for benchmark baselines, where the architecture differs per entry.
    Same discipline as load_trained_model: no fallback, and any missing,
    unexpected or misshapen key is fatal, because a partially loaded
    baseline still produces plausible numbers.
    """
    _add_paths(project_root)
    if ":" not in spec:
        sys.exit("model spec must be 'module.path:ClassName', got %r" % spec)
    mod_name, attr = spec.split(":", 1)
    try:
        mod = __import__(mod_name, fromlist=[attr])
    except ImportError as exc:
        sys.exit("cannot import %s (%r)" % (mod_name, exc))
    cls = getattr(mod, attr, None)
    if cls is None:
        sys.exit("%s has no attribute %r" % (mod_name, attr))

    try:
        model = cls(**(kwargs or {}))
    except Exception as exc:
        sys.exit("could not construct %s(**%r): %r" % (spec, kwargs, exc))

    if checkpoint is None:
        model.eval()
        return model, {"spec": spec, "kwargs": kwargs,
                       "parameters_total": sum(p.numel()
                                               for p in model.parameters())}

    if not os.path.exists(checkpoint):
        sys.exit("checkpoint not found: %s" % checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    recorded_arch = None
    if isinstance(state, dict):
        recorded_arch = state.get("architecture") or state.get("arch")
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break

    # If the checkpoint records which architecture wrote it, hold the caller
    # to it. Two baselines whose state dicts happen to be compatible would
    # otherwise load into each other silently.
    if recorded_arch:
        wanted = spec.split(":", 1)[1]
        if str(recorded_arch) != wanted:
            sys.exit(
                "ABORT: %s records architecture %r but was requested as %r. "
                "Refusing to evaluate a checkpoint under the wrong "
                "architecture label."
                % (checkpoint, recorded_arch, wanted))

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        sys.exit("ABORT: tensor shapes in %s do not match %s(**%r).\n%s"
                 % (checkpoint, spec, kwargs, exc))
    if missing or unexpected:
        sys.exit("ABORT: %s did not load cleanly into %s: %d missing, %d "
                 "unexpected key(s).\n  missing:    %s\n  unexpected: %s"
                 % (checkpoint, spec, len(missing), len(unexpected),
                    list(missing)[:6], list(unexpected)[:6]))
    model.eval()
    meta = {"spec": spec, "kwargs": kwargs, "checkpoint": checkpoint,
            "architecture_recorded": recorded_arch,
            "parameters_total": sum(p.numel() for p in model.parameters())}
    if provenance is not None:
        provenance.setdefault("baseline_models", []).append(meta)
    return model, meta


def load_gradient_methods(model, ig_steps=20, project_root=None,
                          provenance=None):
    """Construct ONLY the two input-space attribution methods.

    Deliberately separate from load_project_explainers: that function
    constructs GradCAM3D, whose output is withdrawn. Nothing here can
    reach it. Confirmed signatures:

        xai_explainers.gradient_methods:VanillaGradient(model)
            .generate(input_tensor, target_class=1) -> [D,H,W] ndarray
        xai_explainers.gradient_methods:IntegratedGradients(model, steps=20)
            .generate(input_tensor, target_class=1) -> [D,H,W] ndarray

    Both return min-max normalised arrays, so the two are directly
    comparable. IntegratedGradients takes its step count at construction.
    """
    root = _add_paths(project_root)
    try:
        from xai_explainers.gradient_methods import (VanillaGradient,
                                                     IntegratedGradients)
    except ImportError as exc:
        sys.exit(
            "Cannot import VanillaGradient / IntegratedGradients from "
            "xai_explainers.gradient_methods (%s).\nLooked under %s and %s."
            % (exc, os.path.join(root, "src"), root))
    try:
        vg = VanillaGradient(model)
    except Exception as exc:
        sys.exit("Could not construct VanillaGradient(model): %r" % (exc,))
    try:
        ig = IntegratedGradients(model, steps=ig_steps)
    except Exception as exc:
        sys.exit("Could not construct IntegratedGradients(model, steps=%d): %r"
                 % (ig_steps, exc))
    for name, obj in (("VanillaGradient", vg), ("IntegratedGradients", ig)):
        if not hasattr(obj, "generate"):
            sys.exit("%s has no generate(); available: %s"
                     % (name, [a for a in dir(obj) if not a.startswith("_")]))

    impl = {
        "vanilla_gradient":
            "xai_explainers.gradient_methods:VanillaGradient.generate(x, cls)",
        "integrated_gradients":
            "xai_explainers.gradient_methods:IntegratedGradients(steps=%d)"
            ".generate(x, cls)" % ig_steps,
    }
    if provenance is not None:
        provenance["implementations"] = impl
        provenance["grad_cam"] = ("excluded: withdrawn, see Section 4.4")
    print("explainers: VanillaGradient, IntegratedGradients(steps=%d). "
          "Grad-CAM is not constructed." % ig_steps, file=sys.stderr)
    return {"vanilla_gradient": lambda x, c: vg.generate(x, c),
            "integrated_gradients": lambda x, c: ig.generate(x, c)}, impl
