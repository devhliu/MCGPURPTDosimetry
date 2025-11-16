"""Geometry processor for material assignment from CT images."""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

from .data_models import GeometryData, MaterialProperties
from ..utils.logging import get_logger


logger = get_logger()


class GeometryProcessor:
    """Processes CT images to create material and density maps.
    
    Converts Hounsfield Unit (HU) values to material IDs and densities
    for Monte Carlo simulation. Supports multi-range HU definitions to
    handle contrast-enhanced CT scans.
    
    Attributes:
        hu_to_material_lut: Lookup table mapping materials to HU ranges
        material_properties: Dictionary of material properties by ID
        material_name_to_id: Mapping from material names to IDs
    """
    
    def __init__(
        self,
        hu_to_material_lut: Dict[str, List[Tuple[float, float]]],
        material_properties: Optional[Dict[int, MaterialProperties]] = None
    ):
        """Initialize GeometryProcessor.
        
        Args:
            hu_to_material_lut: Dictionary mapping material names to list of HU ranges
                Example: {'Soft_Tissue': [(-50, 100)], 'Bone': [(300, 3000)]}
            material_properties: Optional dictionary of material properties by ID
        """
        self.hu_to_material_lut = hu_to_material_lut
        self.material_properties = material_properties or {}
        
        # Create material name to ID mapping
        self.material_name_to_id = {
            name: idx for idx, name in enumerate(sorted(hu_to_material_lut.keys()))
        }
        
        # Create reverse mapping
        self.material_id_to_name = {
            idx: name for name, idx in self.material_name_to_id.items()
        }
        
        logger.info(
            f"GeometryProcessor initialized with {len(self.material_name_to_id)} materials"
        )
        logger.debug(f"Material mapping: {self.material_name_to_id}")
    
    def process_ct_to_materials(
        self,
        ct_tensor: torch.Tensor,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """Convert CT HU values to material IDs.
        
        Args:
            ct_tensor: CT data in Hounsfield Units [X, Y, Z]
            device: Device to perform computation on (defaults to ct_tensor device)
            
        Returns:
            Material ID map [X, Y, Z] with integer material IDs
        """
        if device is None:
            device = ct_tensor.device
        
        logger.info("Converting CT to material map...")
        
        # Initialize material map with -1 (unassigned)
        material_map = torch.full_like(ct_tensor, -1, dtype=torch.int32)
        
        # Assign materials based on HU ranges
        for material_name, hu_ranges in self.hu_to_material_lut.items():
            material_id = self.material_name_to_id[material_name]
            
            # Create mask for all HU ranges of this material
            material_mask = torch.zeros_like(ct_tensor, dtype=torch.bool)
            
            for hu_min, hu_max in hu_ranges:
                range_mask = (ct_tensor >= hu_min) & (ct_tensor <= hu_max)
                material_mask = material_mask | range_mask
            
            # Assign material ID where mask is True
            material_map[material_mask] = material_id
        
        # Check for unassigned voxels
        unassigned_count = torch.sum(material_map == -1).item()
        if unassigned_count > 0:
            total_voxels = material_map.numel()
            logger.warning(
                f"{unassigned_count}/{total_voxels} voxels "
                f"({100*unassigned_count/total_voxels:.2f}%) have no material assignment"
            )
        
        # Log material distribution
        for material_name, material_id in self.material_name_to_id.items():
            count = torch.sum(material_map == material_id).item()
            if count > 0:
                logger.debug(
                    f"Material '{material_name}' (ID={material_id}): {count} voxels"
                )
        
        logger.info("Material map created successfully")
        return material_map
    
    def process_ct_to_densities(
        self,
        ct_tensor: torch.Tensor,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """Convert CT HU values to physical densities.
        
        Uses standard HU-to-density conversion:
        - For HU < 0 (air/lung): ρ = 1.0 + HU/1000
        - For HU >= 0 (soft tissue/bone): ρ = 1.0 + HU/1000
        
        Args:
            ct_tensor: CT data in Hounsfield Units [X, Y, Z]
            device: Device to perform computation on (defaults to ct_tensor device)
            
        Returns:
            Density map in g/cm³ [X, Y, Z]
        """
        if device is None:
            device = ct_tensor.device
        
        logger.info("Converting CT to density map...")
        
        # Standard HU-to-density conversion
        # ρ (g/cm³) = 1.0 + HU/1000
        density_map = 1.0 + ct_tensor / 1000.0
        
        # Clamp to reasonable range (0.001 to 3.0 g/cm³)
        density_map = torch.clamp(density_map, min=0.001, max=3.0)
        
        # Log density statistics
        logger.debug(
            f"Density map statistics: "
            f"min={density_map.min().item():.3f}, "
            f"max={density_map.max().item():.3f}, "
            f"mean={density_map.mean().item():.3f} g/cm³"
        )
        
        logger.info("Density map created successfully")
        return density_map
    
    def create_geometry_data(
        self,
        ct_tensor: torch.Tensor,
        voxel_size: Tuple[float, float, float],
        affine_matrix: np.ndarray,
        mask_dict: Optional[Dict[str, torch.Tensor]] = None,
        mask_priority_order: Optional[List[str]] = None,
        use_ct_density: bool = True
    ) -> GeometryData:
        """Create complete geometry data from CT image with optional masks.
        
        Args:
            ct_tensor: CT data in Hounsfield Units [X, Y, Z]
            voxel_size: Voxel dimensions in mm (dx, dy, dz)
            affine_matrix: 4x4 spatial transform matrix
            mask_dict: Optional dictionary of tissue masks
            mask_priority_order: Optional priority order for overlapping masks
            use_ct_density: If True, use CT-derived density in masked regions
            
        Returns:
            GeometryData object with material and density maps
        """
        logger.info("Creating geometry data...")
        
        # Process CT to materials and densities (with optional mask override)
        if mask_dict is not None and len(mask_dict) > 0:
            material_map, density_map = self.process_with_mask_override(
                ct_tensor, mask_dict, mask_priority_order, use_ct_density
            )
        else:
            material_map = self.process_ct_to_materials(ct_tensor)
            density_map = self.process_ct_to_densities(ct_tensor)
        
        # Get dimensions
        dimensions = tuple(ct_tensor.shape)
        
        # Create GeometryData object
        geometry = GeometryData(
            material_map=material_map,
            density_map=density_map,
            voxel_size=voxel_size,
            dimensions=dimensions,
            affine_matrix=affine_matrix
        )
        
        logger.info(
            f"Geometry data created: {dimensions[0]}x{dimensions[1]}x{dimensions[2]} voxels, "
            f"voxel size={voxel_size} mm"
        )
        
        return geometry
    
    def get_material_properties(self, material_id: int) -> Optional[MaterialProperties]:
        """Get material properties for a given material ID.
        
        Args:
            material_id: Material ID
            
        Returns:
            MaterialProperties object or None if not found
        """
        return self.material_properties.get(material_id)
    
    def add_material_properties(
        self,
        material_name: str,
        density: float,
        composition: Dict[str, float]
    ) -> None:
        """Add material properties for a material.
        
        Args:
            material_name: Name of material
            density: Material density in g/cm³
            composition: Elemental composition {element: mass_fraction}
        """
        if material_name not in self.material_name_to_id:
            logger.warning(
                f"Material '{material_name}' not in HU lookup table"
            )
            return
        
        material_id = self.material_name_to_id[material_name]
        
        self.material_properties[material_id] = MaterialProperties(
            material_id=material_id,
            name=material_name,
            density=density,
            composition=composition
        )
        
        logger.debug(
            f"Added properties for material '{material_name}' (ID={material_id})"
        )
    
    def process_with_mask_override(
        self,
        ct_tensor: torch.Tensor,
        mask_dict: Optional[Dict[str, torch.Tensor]],
        priority_order: Optional[List[str]] = None,
        use_ct_density: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process CT image with mask-based material override.
        
        Args:
            ct_tensor: CT image in HU [X, Y, Z]
            mask_dict: Dictionary of {tissue_name: mask_tensor}, or None
            priority_order: Order for applying overlapping masks (last has highest priority)
            use_ct_density: If True, use CT-derived density; if False, use material database density
            
        Returns:
            Tuple of (material_map, density_map)
        """
        # Step 1: Initialize with HU-based assignment
        material_map = self.process_ct_to_materials(ct_tensor)
        density_map = self.process_ct_to_densities(ct_tensor)
        
        # If no masks, return HU-based maps
        if mask_dict is None or len(mask_dict) == 0:
            return material_map, density_map
        
        logger.info(f"Applying {len(mask_dict)} tissue masks...")
        
        # Step 2: Validate masks
        self._validate_masks(ct_tensor, mask_dict)
        
        # Step 3: Validate and apply masks in priority order
        if priority_order is None:
            priority_order = list(mask_dict.keys())
        else:
            # Validate priority order
            for tissue_name in priority_order:
                if tissue_name not in mask_dict:
                    raise ValueError(
                        f"Tissue '{tissue_name}' in priority_order but not in mask_dict. "
                        f"Available tissues: {list(mask_dict.keys())}"
                    )
        
        for tissue_name in priority_order:
            if tissue_name not in mask_dict:
                logger.warning(f"Tissue {tissue_name} in priority order but not in mask_dict")
                continue
            
            mask = mask_dict[tissue_name]
            
            # Get material ID for this tissue
            if tissue_name not in self.material_name_to_id:
                logger.warning(
                    f"Tissue '{tissue_name}' not in material lookup table, skipping"
                )
                continue
            
            material_id = self.material_name_to_id[tissue_name]
            
            # Apply mask to material map
            material_map[mask > 0] = material_id
            
            # Handle density
            if not use_ct_density:
                # Use fixed material density from database
                if material_id in self.material_properties:
                    material_density = self.material_properties[material_id].density
                    density_map[mask > 0] = material_density
                else:
                    logger.warning(
                        f"No density data for material ID {material_id}, "
                        f"keeping CT-derived density"
                    )
            # else: keep CT-derived density
            
            voxel_count = torch.sum(mask > 0).item()
            logger.debug(
                f"Applied mask for {tissue_name}: {voxel_count} voxels "
                f"assigned to material ID {material_id}"
            )
        
        logger.info("Mask-based material assignment complete")
        
        return material_map, density_map
    
    def validate_mask_compatibility(
        self,
        ct_tensor: torch.Tensor,
        mask_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Validate that masks are compatible with CT image.
        
        Public API for mask validation. Checks that all masks have the same
        dimensions as the CT image and contain valid binary data.
        
        Args:
            ct_tensor: CT image tensor
            mask_dict: Dictionary of tissue masks
            
        Raises:
            ValueError: If masks are incompatible with CT image
            
        Example:
            >>> processor = GeometryProcessor(hu_lut)
            >>> processor.validate_mask_compatibility(ct_tensor, masks)
        """
        ct_shape = ct_tensor.shape
        
        for tissue_name, mask in mask_dict.items():
            # Check dimensions
            if mask.shape != ct_shape:
                raise ValueError(
                    f"Mask '{tissue_name}' shape {mask.shape} does not match "
                    f"CT shape {ct_shape}"
                )
            
            # Check mask values (should be binary or close to it)
            unique_values = torch.unique(mask)
            if len(unique_values) > 2:
                logger.warning(
                    f"Mask '{tissue_name}' has {len(unique_values)} unique values. "
                    f"Treating all non-zero values as mask region."
                )
    
    def _validate_masks(
        self,
        ct_tensor: torch.Tensor,
        mask_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Internal wrapper for validate_mask_compatibility.
        
        Deprecated: Use validate_mask_compatibility() instead.
        """
        self.validate_mask_compatibility(ct_tensor, mask_dict)
    
    @staticmethod
    def get_default_material_properties() -> Dict[str, Dict]:
        """Get default material properties for standard biological tissues.
        
        Returns:
            Dictionary mapping material names to properties
        """
        return {
            'Air': {
                'density': 0.001205,
                'composition': {'N': 0.755, 'O': 0.232, 'Ar': 0.013}
            },
            'Lung': {
                'density': 0.26,
                'composition': {'H': 0.103, 'C': 0.105, 'N': 0.031, 'O': 0.749, 'P': 0.002, 'S': 0.002, 'Cl': 0.003, 'K': 0.002, 'Ca': 0.003}
            },
            'Fat': {
                'density': 0.95,
                'composition': {'H': 0.114, 'C': 0.598, 'N': 0.007, 'O': 0.278, 'S': 0.001, 'Cl': 0.001, 'K': 0.001}
            },
            'Soft_Tissue': {
                'density': 1.04,
                'composition': {'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.708, 'Na': 0.002, 'P': 0.003, 'S': 0.003, 'Cl': 0.002, 'K': 0.003}
            },
            'Soft_Tissue_Contrast': {
                'density': 1.10,
                'composition': {'H': 0.095, 'C': 0.133, 'N': 0.032, 'O': 0.660, 'Na': 0.002, 'P': 0.003, 'S': 0.003, 'Cl': 0.002, 'K': 0.003, 'I': 0.067}
            },
            'Bone_Trabecular': {
                'density': 1.18,
                'composition': {'H': 0.085, 'C': 0.404, 'N': 0.028, 'O': 0.436, 'Mg': 0.002, 'P': 0.034, 'S': 0.001, 'Ca': 0.010}
            },
            'Bone_Cortical': {
                'density': 1.92,
                'composition': {'H': 0.034, 'C': 0.155, 'N': 0.042, 'O': 0.435, 'Mg': 0.002, 'P': 0.103, 'S': 0.003, 'Ca': 0.226}
            },
        }
