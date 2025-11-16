"""Input manager for flexible medical image I/O."""

from typing import Union, Tuple, Dict, Optional
from pathlib import Path
import torch
import nibabel as nib
import numpy as np

from ..utils.validation import (
    validate_nifti_file,
    validate_nifti_object,
    validate_image_compatibility,
    InvalidImageFormatError
)
from ..utils.logging import get_logger
from ..utils.path_utils import validate_path


logger = get_logger()


class InputManager:
    """Manages flexible input of medical images (files or objects).
    
    Supports both file paths and nibabel image objects for CT and activity data.
    Validates spatial compatibility and preserves metadata for output.
    
    Attributes:
        ct_image: Loaded CT nibabel image
        activity_image: Loaded activity nibabel image
        ct_tensor: CT data as PyTorch tensor
        activity_tensor: Activity data as PyTorch tensor
        voxel_size: Voxel dimensions in mm (dx, dy, dz)
        dimensions: Grid dimensions (nx, ny, nz)
        affine_matrix: 4x4 spatial transform matrix
    """
    
    def __init__(self):
        """Initialize InputManager."""
        self.ct_image = None
        self.activity_image = None
        self.ct_tensor = None
        self.activity_tensor = None
        self.voxel_size = None
        self.dimensions = None
        self.affine_matrix = None
    
    def load_ct_image(
        self,
        source: Union[str, nib.Nifti1Image],
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Load CT anatomical image from file or object.
        
        Args:
            source: Either file path (str) or nibabel image object
            device: Device to load tensor to ('cuda' or 'cpu')
            
        Returns:
            CT data as PyTorch tensor [X, Y, Z]
            
        Raises:
            InvalidImageFormatError: If input is not valid NIfTI
        """
        logger.info("Loading CT image...")
        
        # Detect input type and load
        if isinstance(source, str):
            # Validate path for security
            safe_path = validate_path(source, must_exist=True)
            logger.debug(f"Loading CT from file: {safe_path}")
            validate_nifti_file(str(safe_path))
            self.ct_image = nib.load(str(safe_path))
        elif isinstance(source, (nib.Nifti1Image, nib.Nifti2Image)):
            logger.debug("Loading CT from nibabel object")
            validate_nifti_object(source)
            self.ct_image = source
        else:
            raise InvalidImageFormatError(
                f"CT source must be file path (str) or nibabel image, got {type(source)}"
            )
        
        # Extract data and metadata
        ct_data = self.ct_image.get_fdata()
        self.affine_matrix = self.ct_image.affine
        self.dimensions = ct_data.shape
        
        # Calculate voxel size from affine matrix
        self.voxel_size = self._extract_voxel_size(self.affine_matrix)
        
        # Convert to PyTorch tensor
        self.ct_tensor = torch.from_numpy(ct_data.astype(np.float32)).to(device)
        
        logger.info(
            f"CT image loaded: shape={self.dimensions}, "
            f"voxel_size={self.voxel_size} mm"
        )
        
        return self.ct_tensor
    
    def load_activity_image(
        self,
        source: Union[str, nib.Nifti1Image],
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Load activity map from file or object.
        
        Args:
            source: Either file path (str) or nibabel image object
            device: Device to load tensor to ('cuda' or 'cpu')
            
        Returns:
            Activity data as PyTorch tensor [X, Y, Z] in Bq/pixel
            
        Raises:
            InvalidImageFormatError: If input is not valid NIfTI
        """
        logger.info("Loading activity image...")
        
        # Detect input type and load
        if isinstance(source, str):
            # Validate path for security
            safe_path = validate_path(source, must_exist=True)
            logger.debug(f"Loading activity from file: {safe_path}")
            validate_nifti_file(str(safe_path))
            self.activity_image = nib.load(str(safe_path))
        elif isinstance(source, (nib.Nifti1Image, nib.Nifti2Image)):
            logger.debug("Loading activity from nibabel object")
            validate_nifti_object(source)
            self.activity_image = source
        else:
            raise InvalidImageFormatError(
                f"Activity source must be file path (str) or nibabel image, got {type(source)}"
            )
        
        # Extract data
        activity_data = self.activity_image.get_fdata()
        
        # Convert to PyTorch tensor
        self.activity_tensor = torch.from_numpy(activity_data.astype(np.float32)).to(device)
        
        logger.info(
            f"Activity image loaded: shape={activity_data.shape}, "
            f"total_activity={torch.sum(self.activity_tensor).item():.2e} Bq"
        )
        
        return self.activity_tensor
    
    def validate_image_compatibility(self) -> bool:
        """Validate that CT and activity images are spatially compatible.
        
        Returns:
            True if images are compatible
            
        Raises:
            ImageDimensionMismatchError: If images are incompatible
            RuntimeError: If images haven't been loaded yet
        """
        if self.ct_image is None or self.activity_image is None:
            raise RuntimeError(
                "Both CT and activity images must be loaded before validation"
            )
        
        logger.debug("Validating image compatibility...")
        
        ct_shape = self.ct_image.shape
        ct_affine = self.ct_image.affine
        activity_shape = self.activity_image.shape
        activity_affine = self.activity_image.affine
        
        validate_image_compatibility(
            ct_shape, ct_affine,
            activity_shape, activity_affine
        )
        
        logger.info("Image compatibility validated successfully")
        return True
    
    def get_voxel_dimensions(self) -> Tuple[float, float, float]:
        """Get voxel dimensions in mm.
        
        Returns:
            Tuple of (dx, dy, dz) in mm
            
        Raises:
            RuntimeError: If images haven't been loaded yet
        """
        if self.voxel_size is None:
            raise RuntimeError("CT image must be loaded first")
        
        return self.voxel_size
    
    def get_affine_matrix(self) -> np.ndarray:
        """Get affine transformation matrix.
        
        Returns:
            4x4 affine matrix
            
        Raises:
            RuntimeError: If images haven't been loaded yet
        """
        if self.affine_matrix is None:
            raise RuntimeError("CT image must be loaded first")
        
        return self.affine_matrix
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Get grid dimensions.
        
        Returns:
            Tuple of (nx, ny, nz)
            
        Raises:
            RuntimeError: If images haven't been loaded yet
        """
        if self.dimensions is None:
            raise RuntimeError("CT image must be loaded first")
        
        return self.dimensions
    
    def load_segmentation_masks(
        self,
        mask_sources: Union[Dict[str, Union[str, nib.Nifti1Image]], None],
        device: str = 'cuda'
    ) -> Union[Dict[str, torch.Tensor], None]:
        """Load segmentation masks from files or objects.
        
        Args:
            mask_sources: Dictionary mapping tissue names to mask sources
                         (file paths or nibabel objects), or None
            device: Device to load tensors to ('cuda' or 'cpu')
            
        Returns:
            Dictionary mapping tissue names to mask tensors, or None if no masks
            
        Raises:
            InvalidImageFormatError: If mask input is not valid
        """
        if mask_sources is None:
            return None
        
        logger.info(f"Loading {len(mask_sources)} segmentation masks...")
        
        mask_tensors = {}
        
        for tissue_name, source in mask_sources.items():
            # Detect input type and load
            if isinstance(source, str):
                # Validate path for security
                safe_path = validate_path(source, must_exist=True)
                logger.debug(f"Loading mask for {tissue_name} from file: {safe_path}")
                validate_nifti_file(str(safe_path))
                mask_img = nib.load(str(safe_path))
            elif isinstance(source, (nib.Nifti1Image, nib.Nifti2Image)):
                logger.debug(f"Loading mask for {tissue_name} from nibabel object")
                validate_nifti_object(source)
                mask_img = source
            else:
                raise InvalidImageFormatError(
                    f"Mask source for {tissue_name} must be file path (str) or "
                    f"nibabel image, got {type(source)}"
                )
            
            # Extract mask data
            mask_data = mask_img.get_fdata()
            
            # Convert to binary mask (any non-zero value is True)
            mask_binary = (mask_data > 0).astype(np.float32)
            
            # Convert to PyTorch tensor
            mask_tensor = torch.from_numpy(mask_binary).to(device)
            
            mask_tensors[tissue_name] = mask_tensor
            
            voxel_count = torch.sum(mask_tensor).item()
            logger.info(
                f"Mask for {tissue_name} loaded: "
                f"{voxel_count} voxels ({100*voxel_count/mask_tensor.numel():.1f}%)"
            )
        
        return mask_tensors
    
    @staticmethod
    def _extract_voxel_size(affine: np.ndarray) -> Tuple[float, float, float]:
        """Extract voxel size from affine matrix.
        
        Args:
            affine: 4x4 affine transformation matrix
            
        Returns:
            Tuple of (dx, dy, dz) in mm
        """
        # Voxel size is the norm of the first 3 columns (excluding translation)
        dx = np.linalg.norm(affine[:3, 0])
        dy = np.linalg.norm(affine[:3, 1])
        dz = np.linalg.norm(affine[:3, 2])
        
        return (float(dx), float(dy), float(dz))
