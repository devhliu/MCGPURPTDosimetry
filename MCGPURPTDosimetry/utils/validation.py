"""Validation utilities for input data and configuration."""

from pathlib import Path
from typing import Union
import nibabel as nib
import numpy as np

from .config import SimulationConfig
from .logging import get_logger


logger = get_logger()


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class InvalidImageFormatError(ValidationError):
    """Raised when input image format is invalid."""
    pass


class ImageDimensionMismatchError(ValidationError):
    """Raised when CT and activity images have incompatible dimensions."""
    pass


class InvalidConfigurationError(ValidationError):
    """Raised when configuration parameters are invalid."""
    pass


def validate_nifti_file(file_path: str) -> None:
    """Validate that a file is a valid NIfTI file.
    
    Args:
        file_path: Path to NIfTI file
        
    Raises:
        InvalidImageFormatError: If file is not valid NIfTI
    """
    path = Path(file_path)
    
    # Check file exists
    if not path.exists():
        raise InvalidImageFormatError(f"File does not exist: {file_path}")
    
    # Check file extension
    valid_extensions = ['.nii', '.nii.gz']
    if not any(str(path).endswith(ext) for ext in valid_extensions):
        raise InvalidImageFormatError(
            f"File must have .nii or .nii.gz extension, got: {file_path}"
        )
    
    # Try to load with nibabel
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Check that data is 3D
        if data.ndim != 3:
            raise InvalidImageFormatError(
                f"Image must be 3D, got {data.ndim}D: {file_path}"
            )
        
        # Check for valid data
        if np.all(np.isnan(data)):
            raise InvalidImageFormatError(
                f"Image contains only NaN values: {file_path}"
            )
            
    except Exception as e:
        if isinstance(e, InvalidImageFormatError):
            raise
        raise InvalidImageFormatError(
            f"Failed to load NIfTI file {file_path}: {str(e)}"
        )


def validate_nifti_object(img: nib.Nifti1Image) -> None:
    """Validate that an object is a valid nibabel NIfTI image.
    
    Args:
        img: Nibabel image object
        
    Raises:
        InvalidImageFormatError: If object is not valid NIfTI
    """
    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        raise InvalidImageFormatError(
            f"Object must be nibabel Nifti1Image or Nifti2Image, got {type(img)}"
        )
    
    try:
        data = img.get_fdata()
        
        # Check that data is 3D
        if data.ndim != 3:
            raise InvalidImageFormatError(
                f"Image must be 3D, got {data.ndim}D"
            )
        
        # Check for valid data
        if np.all(np.isnan(data)):
            raise InvalidImageFormatError(
                "Image contains only NaN values"
            )
            
    except Exception as e:
        if isinstance(e, InvalidImageFormatError):
            raise
        raise InvalidImageFormatError(
            f"Failed to validate NIfTI object: {str(e)}"
        )


def validate_image_compatibility(
    ct_shape: tuple,
    ct_affine: np.ndarray,
    activity_shape: tuple,
    activity_affine: np.ndarray,
    tolerance: float = 1e-3
) -> None:
    """Validate that CT and activity images are spatially compatible.
    
    Args:
        ct_shape: CT image shape (nx, ny, nz)
        ct_affine: CT affine matrix (4x4)
        activity_shape: Activity image shape (nx, ny, nz)
        activity_affine: Activity affine matrix (4x4)
        tolerance: Tolerance for affine matrix comparison
        
    Raises:
        ImageDimensionMismatchError: If images are incompatible
    """
    # Check dimensions match
    if ct_shape != activity_shape:
        raise ImageDimensionMismatchError(
            f"CT and activity images have different dimensions: "
            f"CT={ct_shape}, Activity={activity_shape}"
        )
    
    # Check affine matrices match (within tolerance)
    affine_diff = np.abs(ct_affine - activity_affine)
    if np.any(affine_diff > tolerance):
        max_diff = np.max(affine_diff)
        raise ImageDimensionMismatchError(
            f"CT and activity images have different spatial orientations. "
            f"Maximum affine difference: {max_diff:.6f} (tolerance: {tolerance})"
        )
    
    logger.debug(
        f"Image compatibility validated: shape={ct_shape}, "
        f"max affine diff={np.max(affine_diff):.6e}"
    )


def validate_config(config: SimulationConfig) -> None:
    """Validate simulation configuration.
    
    Args:
        config: Simulation configuration
        
    Raises:
        InvalidConfigurationError: If configuration is invalid
    """
    try:
        # Configuration validation is done in __post_init__
        # This function can be extended for additional runtime checks
        
        # Check database files exist if paths are provided
        if config.decay_database_path:
            db_path = Path(config.decay_database_path)
            if not db_path.exists():
                logger.warning(
                    f"Decay database not found: {config.decay_database_path}"
                )
        
        if config.cross_section_database_path:
            xs_path = Path(config.cross_section_database_path)
            if not xs_path.exists():
                logger.warning(
                    f"Cross-section database not found: {config.cross_section_database_path}"
                )
        
        # Check CUDA availability if device is 'cuda'
        if config.device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    raise InvalidConfigurationError(
                        "CUDA device requested but CUDA is not available. "
                        "Set device='cpu' or install CUDA support."
                    )
            except ImportError:
                raise InvalidConfigurationError(
                    "PyTorch not installed. Please install torch."
                )
        
        logger.debug("Configuration validation passed")
        
    except Exception as e:
        if isinstance(e, InvalidConfigurationError):
            raise
        raise InvalidConfigurationError(f"Configuration validation failed: {str(e)}")
