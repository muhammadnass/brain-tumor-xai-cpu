"""
Configuration management for neurosurgery XAI system
File location: src/config.py
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SEGMENTATION MODEL CONFIG
# ============================================================================
class SegmentationConfig:
    """3D U-Net segmentation model configuration"""
    
    # Model architecture
    model_name = "unet_3d"
    input_channels = 4  # T1, T2, FLAIR, T1c
    output_channels = 3  # background, tumor, edema
    image_size = (128, 128, 128)
    
    # Training
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 100
    weight_decay = 1e-5
    
    # Device
    device = "cuda"  # or "cpu"
    num_workers = 4
    
    # Checkpoint
    checkpoint = MODELS_DIR / "segmentation" / "unet_3d.pth"
    save_frequency = 10  # Save every N epochs
    
    # Data augmentation
    use_augmentation = True
    augmentation_probability = 0.5

# ============================================================================
# CLASSIFICATION MODEL CONFIG
# ============================================================================
class ClassificationConfig:
    """2D ResNet classification model configuration"""
    
    # Model architecture
    model_name = "resnet50"
    num_classes = 3  # glioblastoma, meningioma, metastases
    input_size = (224, 224)
    pretrained = True
    
    # Training
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    weight_decay = 1e-5
    
    # Device
    device = "cuda"
    num_workers = 4
    
    # Checkpoint
    checkpoint = MODELS_DIR / "classification" / "resnet50.pth"
    save_frequency = 5

# ============================================================================
# XAI CONFIGURATION
# ============================================================================
class XAIConfig:
    """Explainable AI configuration"""
    
    # Grad-CAM
    use_grad_cam = True
    grad_cam_layer = "layer4"
    grad_cam_threshold = 0.7
    
    # SHAP
    use_shap = True
    shap_sample_size = 100
    
    # LIME
    use_lime = True
    lime_num_samples = 1000
    
    # Confidence mapping (MC Dropout)
    use_mc_dropout = True
    mc_samples = 10
    dropout_rate = 0.5
    
    # Visualizations
    colormap = "jet"  # matplotlib colormap
    overlay_alpha = 0.4

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
BRATS_DATASET = DATA_DIR / "brats_dataset"
SAMPLE_CASES = DATA_DIR / "sample_cases"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset splits
DATASET_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
}

# ============================================================================
# FLASK API CONFIGURATION
# ============================================================================
class APIConfig:
    """Flask API configuration"""
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 5000
    TESTING = False
    
    # File upload
    UPLOAD_FOLDER = DATA_DIR / "uploads"
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {'nii', 'gz', 'dcm'}
    
    # API
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
class StreamlitConfig:
    """Streamlit UI configuration"""
    api_url = "http://localhost:5000/api"
    show_explanations = True
    debug_mode = True
    max_file_size_mb = 500

# ============================================================================
# USER STUDY CONFIGURATION
# ============================================================================
class UserStudyConfig:
    """Configuration for neurosurgeon user studies"""
    
    # Study parameters
    num_participants = 20
    test_cases_per_participant = 10
    questionnaire_version = "v1.0"
    likert_scale = 5  # 1-5 scale
    
    # Timing
    session_duration_minutes = 120
    case_review_time_minutes = 8
    
    # Output
    results_dir = RESULTS_DIR / "user_study_results"

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================
class PreprocessingConfig:
    """Data preprocessing configuration"""
    
    # Normalization
    normalize_type = "z_score"  # or "minmax"
    clip_outliers = True
    outlier_percentile = 0.5
    
    # Resizing
    target_shape = (128, 128, 128)
    interpolation_order = 3
    
    # Caching
    use_cache = True
    cache_dir = PROCESSED_DATA_DIR / "cache"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
class LoggingConfig:
    """Logging configuration"""
    
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR = LOGS_DIR
    LOG_FILE = LOGS_DIR / "neurosurgery_xai.log"
    
    # Format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# EVALUATION METRICS CONFIGURATION
# ============================================================================
EVALUATION_METRICS = {
    "segmentation": {
        "metrics": ["dice", "jaccard", "hausdorff", "sensitivity", "specificity", "precision"],
        "dice_threshold": 0.85,
        "hausdorff_threshold": 5.0  # mm
    },
    "classification": {
        "metrics": ["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"],
        "accuracy_threshold": 0.90
    },
    "xai": {
        "metrics": ["fidelity", "interpretability_score", "user_comprehension", "explanation_stability"],
        "fidelity_threshold": 0.80
    },
    "surgical_planning": {
        "metrics": ["planning_time", "surgeon_confidence", "plan_accuracy", "adoption_intention"],
        "confidence_threshold": 4.0  # Likert scale
    }
}

# ============================================================================
# TUMOR CLASSIFICATION LABELS
# ============================================================================
TUMOR_CLASSES = {
    0: "glioblastoma",
    1: "meningioma",
    2: "metastases"
}

TUMOR_CLASS_COLORS = {
    0: "#FF0000",  # Red for glioblastoma
    1: "#00FF00",  # Green for meningioma
    2: "#0000FF"   # Blue for metastases
}

# ============================================================================
# MRI MODALITIES
# ============================================================================
MRI_MODALITIES = {
    "T1": "T1-weighted",
    "T2": "T2-weighted",
    "FLAIR": "FLAIR (Fluid Attenuated Inversion Recovery)",
    "T1c": "T1-weighted with contrast"
}

# ============================================================================
# SURGICAL PLANNING PARAMETERS
# ============================================================================
class SurgicalPlanningConfig:
    """Surgical planning configuration"""
    
    # Safety margins (in mm)
    default_margin = 5
    margin_for_high_grade = 10  # Aggressive tumors
    margin_for_low_grade = 3    # Conservative approach
    
    # Risk assessment
    proximity_to_ventricles_threshold = 10  # mm
    proximity_to_critical_areas_threshold = 5  # mm
    
    # Complexity scoring
    complexity_thresholds = {
        "simple": {"volume_max": 20, "proximity_min": 20},  # cm³ and mm
        "moderate": {"volume_max": 50, "proximity_min": 10},
        "complex": {"volume_max": float('inf'), "proximity_min": 0}
    }

# ============================================================================
# FUNCTION TO GET CONFIG
# ============================================================================
def get_config(config_name: str):
    """Get configuration by name"""
    configs = {
        "segmentation": SegmentationConfig,
        "classification": ClassificationConfig,
        "xai": XAIConfig,
        "api": APIConfig,
        "streamlit": StreamlitConfig,
        "user_study": UserStudyConfig,
        "preprocessing": PreprocessingConfig,
        "logging": LoggingConfig,
        "surgical_planning": SurgicalPlanningConfig
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

if __name__ == "__main__":
    print("✅ Configuration loaded successfully")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
