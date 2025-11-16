"""Dose synthesis and uncertainty calculation module."""

from typing import Dict, Union, Optional
from pathlib import Path
import torch
import numpy as np
import nibabel as nib

from .data_models import GeometryData
from ..utils.logging import get_logger


logger = get_logger()


class DoseSynthesis:
    """Synthesizes dose contributions and calculates uncertainties.
    
    Combines dose maps from multiple nuclides, calculates statistical
    uncertainties using the batch method, and exports results in NIfTI format.
    
    Attributes:
        geometry: Geometry data
        num_batches: Number of batches for uncertainty calculation
        nuclide_doses: Dictionary of dose maps by nuclide
        batch_doses: List of dose maps by batch
        device: Computation device
    """
    
    def __init__(self, geometry: GeometryData, num_batches: int = 10):
        """Initialize DoseSynthesis.
        
        Args:
            geometry: Geometry data
            num_batches: Number of batches for uncertainty calculation
        """
        self.geometry = geometry
        self.num_batches = num_batches
        self.nuclide_doses: Dict[str, torch.Tensor] = {}
        self.batch_doses: list = []
        self.device = geometry.material_map.device
        
        logger.info(
            f"DoseSynthesis initialized with {num_batches} batches"
        )
    
    def accumulate_nuclide_dose(
        self,
        nuclide: str,
        dose_tensor: torch.Tensor
    ) -> None:
        """Accumulate dose contribution from a nuclide.
        
        Args:
            nuclide: Nuclide name
            dose_tensor: Dose map in Gy [X, Y, Z]
        """
        if nuclide in self.nuclide_doses:
            self.nuclide_doses[nuclide] += dose_tensor
        else:
            self.nuclide_doses[nuclide] = dose_tensor.clone()
        
        total_dose = torch.sum(dose_tensor).item()
        logger.info(
            f"Accumulated dose for {nuclide}: {total_dose:.2e} Gy"
        )
    
    def accumulate_batch_dose(self, dose_tensor: torch.Tensor) -> None:
        """Accumulate dose from a batch for uncertainty calculation.
        
        Args:
            dose_tensor: Dose map from batch [X, Y, Z]
        """
        self.batch_doses.append(dose_tensor.clone())
        logger.debug(f"Accumulated batch {len(self.batch_doses)}/{self.num_batches}")
    
    def calculate_total_dose(self) -> torch.Tensor:
        """Calculate total dose from all nuclides.
        
        Returns:
            Total dose map in Gy [X, Y, Z]
        """
        if not self.nuclide_doses:
            logger.warning("No nuclide doses accumulated")
            return torch.zeros(
                self.geometry.dimensions,
                dtype=torch.float32,
                device=self.device
            )
        
        total_dose = torch.zeros(
            self.geometry.dimensions,
            dtype=torch.float32,
            device=self.device
        )
        
        for nuclide, dose in self.nuclide_doses.items():
            total_dose += dose
        
        total_dose_sum = torch.sum(total_dose).item()
        logger.info(f"Total dose calculated: {total_dose_sum:.2e} Gy")
        
        return total_dose
    
    def calculate_uncertainty(self) -> torch.Tensor:
        """Calculate statistical uncertainty using batch method.
        
        Returns:
            Relative standard error map in percent [X, Y, Z]
        """
        if len(self.batch_doses) < 2:
            logger.warning(
                f"Insufficient batches for uncertainty calculation: "
                f"{len(self.batch_doses)}"
            )
            return torch.zeros(
                self.geometry.dimensions,
                dtype=torch.float32,
                device=self.device
            )
        
        logger.info("Calculating uncertainty using batch method...")
        
        # Stack batch doses
        batch_stack = torch.stack(self.batch_doses, dim=0)  # [B, X, Y, Z]
        
        # Calculate mean dose
        mean_dose = torch.mean(batch_stack, dim=0)
        
        # Calculate standard deviation
        std_dose = torch.std(batch_stack, dim=0, unbiased=True)
        
        # Calculate relative standard error (%)
        # Avoid division by zero
        rse = torch.zeros_like(mean_dose)
        nonzero_mask = mean_dose > 0
        rse[nonzero_mask] = (std_dose[nonzero_mask] / mean_dose[nonzero_mask]) * 100.0
        
        # Log statistics
        mean_rse = torch.mean(rse[nonzero_mask]).item() if torch.any(nonzero_mask) else 0
        max_rse = torch.max(rse).item()
        
        logger.info(
            f"Uncertainty calculated: mean RSE={mean_rse:.2f}%, max RSE={max_rse:.2f}%"
        )
        
        return rse
    
    def export_results(
        self,
        output_format: str = 'file',
        output_path: Optional[str] = None
    ) -> Union[Dict[str, str], Dict[str, nib.Nifti1Image]]:
        """Export dose maps and uncertainty.
        
        Args:
            output_format: 'file' or 'object'
            output_path: Output directory path (required for 'file' format)
            
        Returns:
            Dictionary of file paths (file format) or nibabel objects (object format)
        """
        logger.info(f"Exporting results in {output_format} format...")
        
        # Calculate total dose and uncertainty
        total_dose = self.calculate_total_dose()
        uncertainty = self.calculate_uncertainty()
        
        # Convert to numpy for NIfTI
        total_dose_np = total_dose.cpu().numpy()
        uncertainty_np = uncertainty.cpu().numpy()
        
        # Create NIfTI images
        affine = self.geometry.affine_matrix
        
        total_dose_img = nib.Nifti1Image(total_dose_np, affine)
        uncertainty_img = nib.Nifti1Image(uncertainty_np, affine)
        
        # Individual nuclide doses
        nuclide_imgs = {}
        for nuclide, dose in self.nuclide_doses.items():
            dose_np = dose.cpu().numpy()
            nuclide_imgs[nuclide] = nib.Nifti1Image(dose_np, affine)
        
        if output_format == 'file':
            if not output_path:
                raise ValueError("output_path required for file format")
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save files
            results = {}
            
            total_dose_path = output_dir / 'total_dose.nii.gz'
            nib.save(total_dose_img, str(total_dose_path))
            results['total_dose'] = str(total_dose_path)
            logger.info(f"Saved total dose: {total_dose_path}")
            
            uncertainty_path = output_dir / 'uncertainty.nii.gz'
            nib.save(uncertainty_img, str(uncertainty_path))
            results['uncertainty'] = str(uncertainty_path)
            logger.info(f"Saved uncertainty: {uncertainty_path}")
            
            for nuclide, img in nuclide_imgs.items():
                nuclide_path = output_dir / f'{nuclide}_dose.nii.gz'
                nib.save(img, str(nuclide_path))
                results[f'{nuclide}_dose'] = str(nuclide_path)
                logger.info(f"Saved {nuclide} dose: {nuclide_path}")
            
            return results
        
        else:  # object format
            results = {
                'total_dose': total_dose_img,
                'uncertainty': uncertainty_img
            }
            
            for nuclide, img in nuclide_imgs.items():
                results[f'{nuclide}_dose'] = img
            
            logger.info("Results returned as nibabel objects")
            return results
    
    def get_dose_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get dose statistics for all nuclides.
        
        Returns:
            Dictionary of statistics by nuclide
        """
        stats = {}
        
        for nuclide, dose in self.nuclide_doses.items():
            dose_np = dose.cpu().numpy()
            nonzero_mask = dose_np > 0
            
            if np.any(nonzero_mask):
                stats[nuclide] = {
                    'total': float(np.sum(dose_np)),
                    'mean': float(np.mean(dose_np[nonzero_mask])),
                    'max': float(np.max(dose_np)),
                    'min': float(np.min(dose_np[nonzero_mask])),
                    'std': float(np.std(dose_np[nonzero_mask]))
                }
            else:
                stats[nuclide] = {
                    'total': 0.0,
                    'mean': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'std': 0.0
                }
        
        # Total dose statistics
        total_dose = self.calculate_total_dose()
        total_dose_np = total_dose.cpu().numpy()
        nonzero_mask = total_dose_np > 0
        
        if np.any(nonzero_mask):
            stats['total'] = {
                'total': float(np.sum(total_dose_np)),
                'mean': float(np.mean(total_dose_np[nonzero_mask])),
                'max': float(np.max(total_dose_np)),
                'min': float(np.min(total_dose_np[nonzero_mask])),
                'std': float(np.std(total_dose_np[nonzero_mask]))
            }
        else:
            stats['total'] = {
                'total': 0.0,
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'std': 0.0
            }
        
        return stats
