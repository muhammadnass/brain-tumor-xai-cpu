"""
MRI data preprocessing utilities
File location: src/data_processing/preprocessor.py
"""
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class MRIPreprocessor:
    """MRI data preprocessing"""
    
    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        normalize_type: str = 'z_score'
    ):
        self.target_shape = target_shape
        self.normalize_type = normalize_type
    
    @staticmethod
    def load_nifti(filepath: str) -> Optional[np.ndarray]:
        """
        Load NIfTI format MRI file
        
        Args:
            filepath: Path to NIfTI file
        
        Returns:
            data: 3D numpy array
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            logger.info(f"✅ Loaded NIfTI: {filepath}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading NIfTI {filepath}: {e}")
            return None
    
    @staticmethod
    def load_dicom(filepath: str) -> Optional[np.ndarray]:
        """
        Load DICOM format MRI file
        
        Args:
            filepath: Path to DICOM file
        
        Returns:
            data: 3D numpy array
        """
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            image = reader.Execute()
            data = sitk.GetArrayFromImage(image)
            logger.info(f"✅ Loaded DICOM: {filepath}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading DICOM {filepath}: {e}")
            return None
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize MRI intensity values

        IMPORTANT: Must match the normalization used during training
        (MONAI's NormalizeIntensityd with nonzero=True) - i.e., z-score
        statistics computed ONLY from non-zero (non-background) voxels.
        Using all voxels (including the large zero-valued skull-stripped
        background) produces a severely different distribution than what
        the model was trained on, causing systematic over-segmentation
        at inference time regardless of the actual input case. This bug
        was discovered via a live-app vs. held-out-evaluation discrepancy:
        the same case scored well (Dice=0.792) under correct MONAI
        preprocessing but predicted >50x too much tumor volume under the
        previous (all-voxel) normalization used here.

        Args:
            data: Input MRI data

        Returns:
            normalized_data: Normalized to mean=0, std=1 (nonzero voxels only)
        """
        if self.normalize_type == 'z_score':
            nonzero_mask = data != 0
            if nonzero_mask.sum() > 0:
                mean = data[nonzero_mask].mean()
                std = data[nonzero_mask].std()
            else:
                mean, std = 0.0, 1.0

            if std > 0:
                normalized = (data - mean) / std
                # Match MONAI behavior: leave background (originally zero) at 0
                # after normalization, rather than shifting it to -mean/std
                normalized = np.where(nonzero_mask, normalized, 0.0)
            else:
                normalized = data
        elif self.normalize_type == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = data
        else:
            normalized = data

        return normalized
    
    def resize(self, data: np.ndarray) -> np.ndarray:
        """
        Resize 3D data to target shape
        
        Args:
            data: Input 3D array
        
        Returns:
            resized: Resized to target_shape
        """
        from scipy import ndimage
        
        # Calculate scaling factors
        current_shape = data.shape
        zoom_factors = [t / c for t, c in zip(self.target_shape, current_shape)]
        
        # Resize using scipy
        resized = ndimage.zoom(data, zoom_factors, order=3)  # Cubic interpolation
        
        return resized.astype(np.float32)
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw MRI data
        
        Returns:
            preprocessed: Preprocessed data
        """
        # Normalize intensity
        normalized = self.normalize(data)
        
        # Resize to target shape
        resized = self.resize(normalized)
        
        return resized


def preprocess_mri_files(
    file_paths: Dict[str, str],
    target_shape: Tuple[int, int, int] = (128, 128, 128)
) -> Optional[np.ndarray]:
    """
    Preprocess multi-modal MRI files
    
    Args:
        file_paths: Dict with keys 'T1', 'T2', 'FLAIR', 'T1c' and file paths as values
        target_shape: Target shape for resizing

    Returns:
        mri_data: Preprocessed 4D array [4, D, H, W]

    IMPORTANT - CHANNEL ORDER: Must exactly match the channel order used
    during training, which follows the Medical Segmentation Decathlon
    Task01_BrainTumour native spec: channel 0=FLAIR, 1=T1, 2=T1c (T1-Gd),
    3=T2 (see scripts/split_decathlon_case.py's MODALITY_CHANNELS mapping,
    which was derived from the original combined 4-channel Decathlon files
    that train_model.py loads directly). Previously this function stacked
    channels as [T1, T2, FLAIR, T1c] - a completely different order that
    fed each modality into the wrong learned channel position, causing
    severe prediction errors independent of (and in addition to) the
    normalization mismatch fixed separately in normalize().
    """
    try:
        preprocessor = MRIPreprocessor(target_shape=target_shape)

        # Order matches training exactly: FLAIR, T1, T1c, T2
        modalities = ['FLAIR', 'T1', 'T1c', 'T2']
        mri_data_list = []
        
        for modality in modalities:
            if modality not in file_paths:
                logger.error(f"Missing modality: {modality}")
                return None
            
            filepath = file_paths[modality]
            
            # Load file
            if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
                data = preprocessor.load_nifti(filepath)
            elif filepath.endswith('.dcm'):
                data = preprocessor.load_dicom(filepath)
            else:
                logger.error(f"Unsupported file format: {filepath}")
                return None
            
            if data is None:
                return None
            
            # Preprocess
            processed = preprocessor.preprocess(data)
            mri_data_list.append(processed)
        
        # Stack modalities: [4, D, H, W]
        mri_data = np.stack(mri_data_list, axis=0)
        
        logger.info(f"✅ Preprocessing complete. Output shape: {mri_data.shape}")
        return mri_data
        
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {e}")
        return None


def postprocess_segmentation(
    segmentation: np.ndarray,
    threshold: float = 0.5,
    apply_morphology: bool = True
) -> np.ndarray:
    """
    Post-process segmentation predictions
    
    Args:
        segmentation: Raw model output [D, H, W]
        threshold: Probability threshold
        apply_morphology: Apply morphological operations
    
    Returns:
        postprocessed: Binary mask
    """
    from scipy import ndimage
    
    # Threshold
    binary = (segmentation > threshold).astype(np.uint8)
    
    if apply_morphology:
        # Remove small connected components
        labeled, num_features = ndimage.label(binary)
        if num_features > 1:
            sizes = ndimage.sum(binary, labeled, range(num_features + 1))
            # Keep only largest component
            largest_label = np.argmax(sizes)
            binary = (labeled == largest_label).astype(np.uint8)
        
        # Morphological closing (fill holes)
        binary = ndimage.binary_closing(binary, iterations=2).astype(np.uint8)
    
    return binary


if __name__ == "__main__":
    print("✅ Preprocessing module loaded successfully")
