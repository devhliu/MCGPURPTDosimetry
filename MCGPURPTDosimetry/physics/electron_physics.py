"""Detailed electron transport physics using Condensed History."""

import torch
import numpy as np
from typing import Tuple, Optional

from ..utils.logging import get_logger
from .constants import (
    ELECTRON_REST_MASS_KEV,
    EPSILON,
    MAX_ENERGY_LOSS_FRACTION,
    STRAGGLING_FACTOR,
    HIGHLAND_CONSTANT,
    BREMSSTRAHLUNG_THRESHOLD_KEV,
    DELTA_RAY_THRESHOLD_KEV,
    RADIATION_LENGTH_CONSTANT,
    RADIATION_LENGTH_LOG_CONSTANT
)


logger = get_logger()


class ElectronPhysics:
    """Detailed electron transport using Condensed History method.
    
    Implements accurate electron transport including:
    - Continuous Slowing Down Approximation (CSDA)
    - Multiple Coulomb Scattering (Goudsmit-Saunderson)
    - Bremsstrahlung photon generation
    - Delta-ray production
    - Range straggling
    
    Attributes:
        device: Computation device
        electron_rest_mass_keV: Electron rest mass energy
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize ElectronPhysics.
        
        Args:
            device: Computation device
        """
        self.device = device
        self.electron_rest_mass_keV = ELECTRON_REST_MASS_KEV
        logger.debug("ElectronPhysics initialized")
    
    def calculate_step_size(
        self,
        energies: torch.Tensor,
        stopping_power: torch.Tensor,
        density: torch.Tensor,
        voxel_size: float,
        max_energy_loss_fraction: float = MAX_ENERGY_LOSS_FRACTION
    ) -> torch.Tensor:
        """Calculate condensed history step size.
        
        Args:
            energies: Electron energies [N]
            stopping_power: Total stopping power in MeV cm²/g [N]
            density: Material density in g/cm³ [N]
            voxel_size: Voxel size in cm
            max_energy_loss_fraction: Maximum fractional energy loss per step
            
        Returns:
            Step sizes in cm [N]
        """
        # Energy-limited step: limit energy loss to fraction of kinetic energy
        # ΔE = S * ρ * Δs
        # Δs = (f * E) / (S * ρ)
        energy_limited_step = (max_energy_loss_fraction * energies) / (
            stopping_power * density * 1000.0 + EPSILON  # Convert MeV to keV
        )
        
        # Geometry-limited step: don't exceed voxel boundary
        geometry_limited_step = torch.full_like(energies, voxel_size)
        
        # Take minimum
        step_size = torch.min(energy_limited_step, geometry_limited_step)
        
        # Ensure positive and reasonable
        step_size = torch.clamp(step_size, min=1e-6, max=voxel_size)
        
        return step_size
    
    def calculate_energy_loss(
        self,
        energies: torch.Tensor,
        step_sizes: torch.Tensor,
        stopping_power_col: torch.Tensor,
        stopping_power_rad: torch.Tensor,
        density: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate energy loss using CSDA.
        
        Args:
            energies: Electron energies [N]
            step_sizes: Step sizes in cm [N]
            stopping_power_col: Collisional stopping power in MeV cm²/g [N]
            stopping_power_rad: Radiative stopping power in MeV cm²/g [N]
            density: Material density in g/cm³ [N]
            
        Returns:
            Tuple of (total_energy_loss, radiative_energy_loss) in keV
        """
        # Total stopping power
        stopping_power_total = stopping_power_col + stopping_power_rad
        
        # Energy loss: ΔE = S * ρ * Δs
        total_energy_loss = stopping_power_total * density * step_sizes * 1000.0  # MeV to keV
        
        # Radiative energy loss (for bremsstrahlung)
        radiative_energy_loss = stopping_power_rad * density * step_sizes * 1000.0
        
        # Apply energy straggling (Landau-Vavilov distribution)
        # Simplified: add Gaussian fluctuation
        straggling_sigma = torch.sqrt(total_energy_loss * STRAGGLING_FACTOR)
        straggling = torch.randn_like(total_energy_loss) * straggling_sigma
        total_energy_loss = total_energy_loss + straggling
        
        # Ensure energy loss doesn't exceed kinetic energy
        # Use torch.minimum/maximum for tensor-tensor operations
        total_energy_loss = torch.maximum(total_energy_loss, torch.zeros_like(total_energy_loss))
        total_energy_loss = torch.minimum(total_energy_loss, energies * 0.95)
        radiative_energy_loss = torch.maximum(radiative_energy_loss, torch.zeros_like(radiative_energy_loss))
        radiative_energy_loss = torch.minimum(radiative_energy_loss, total_energy_loss)
        
        return total_energy_loss, radiative_energy_loss
    
    def calculate_angular_deflection(
        self,
        energies: torch.Tensor,
        step_sizes: torch.Tensor,
        material_z: float,
        density: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate angular deflection using multiple scattering theory.
        
        Uses Goudsmit-Saunderson theory (simplified to Highland formula).
        
        Args:
            energies: Electron energies [N]
            step_sizes: Step sizes in cm [N]
            material_z: Effective atomic number
            density: Material density in g/cm³ [N]
            
        Returns:
            Tuple of (theta, phi) angles in radians
        """
        # Highland formula for multiple scattering
        # θ₀ = (13.6 MeV / βcp) * z * sqrt(x/X₀) * [1 + 0.038 ln(x/X₀)]
        
        # Relativistic parameters
        beta = self._calculate_beta(energies)
        momentum = energies * beta  # Approximate
        
        # Radiation length (simplified)
        radiation_length = self._get_radiation_length(material_z, density)
        
        # Path length in radiation lengths
        x_over_X0 = step_sizes / radiation_length
        
        # Highland formula
        theta_0 = (HIGHLAND_CONSTANT / momentum) * torch.sqrt(x_over_X0) * (
            1.0 + 0.038 * torch.log(x_over_X0 + EPSILON)
        )
        
        # Sample scattering angles from Gaussian (simplified)
        # Full implementation would use Goudsmit-Saunderson distribution
        theta = torch.abs(torch.randn_like(theta_0)) * theta_0
        phi = 2 * np.pi * torch.rand_like(theta)
        
        return theta, phi
    
    def apply_angular_deflection(
        self,
        directions: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """Apply angular deflection to direction vectors.
        
        Args:
            directions: Current directions [N, 3]
            theta: Polar deflection angles [N]
            phi: Azimuthal angles [N]
            
        Returns:
            New directions [N, 3]
        """
        # Convert to spherical coordinates
        # Current direction is z-axis in local frame
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        # Local frame deflection
        local_x = sin_theta * cos_phi
        local_y = sin_theta * sin_phi
        local_z = cos_theta
        
        # Transform to global frame
        # This is simplified; full implementation would use proper rotation matrices
        new_directions = torch.zeros_like(directions)
        
        # Extract current direction components
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
        
        # Simplified rotation (assumes small angles)
        new_directions[:, 0] = dx * cos_theta - dx * sin_theta * cos_phi
        new_directions[:, 1] = dy * cos_theta - dy * sin_theta * sin_phi
        new_directions[:, 2] = dz * cos_theta + sin_theta
        
        # Normalize
        norm = torch.sqrt(torch.sum(new_directions ** 2, dim=1, keepdim=True))
        new_directions = new_directions / (norm + 1e-30)
        
        return new_directions
    
    def sample_bremsstrahlung(
        self,
        energies: torch.Tensor,
        radiative_energy_loss: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample bremsstrahlung photon production.
        
        Args:
            energies: Electron energies [N]
            radiative_energy_loss: Radiative energy loss [N]
            positions: Electron positions [N, 3]
            directions: Electron directions [N, 3]
            
        Returns:
            Tuple of (photon_positions, photon_energies, photon_directions)
        """
        # Determine which electrons produce bremsstrahlung photons
        # Simplified: produce photon if radiative loss > threshold
        produce_photon = radiative_energy_loss > BREMSSTRAHLUNG_THRESHOLD_KEV
        
        n_photons = torch.sum(produce_photon).item()
        
        if n_photons == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros(0, device=self.device),
                torch.zeros((0, 3), device=self.device)
            )
        
        # Photon positions (same as electron)
        photon_positions = positions[produce_photon]
        
        # Photon energies (sample from bremsstrahlung spectrum)
        # Simplified: uniform distribution up to radiative loss
        max_energies = radiative_energy_loss[produce_photon]
        photon_energies = torch.rand(n_photons, device=self.device) * max_energies
        
        # Photon directions (forward-peaked)
        # Sample from dipole distribution
        cos_theta = self._sample_bremsstrahlung_angle(
            energies[produce_photon], photon_energies
        )
        phi = 2 * np.pi * torch.rand(n_photons, device=self.device)
        
        # Rotate electron direction
        photon_directions = self._rotate_direction(
            directions[produce_photon], cos_theta, phi
        )
        
        return photon_positions, photon_energies, photon_directions
    
    def sample_delta_rays(
        self,
        energies: torch.Tensor,
        positions: torch.Tensor,
        delta_ray_threshold: float = DELTA_RAY_THRESHOLD_KEV
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample delta-ray (knock-on electron) production.
        
        Args:
            energies: Electron energies [N]
            positions: Electron positions [N, 3]
            delta_ray_threshold: Minimum energy for delta-ray production
            
        Returns:
            Tuple of (delta_ray_positions, delta_ray_energies)
        """
        # Møller cross-section for electron-electron scattering
        # Simplified: sample based on energy
        
        # Probability of delta-ray production (simplified)
        prob = torch.clamp(energies / 1000.0, max=0.1)  # Higher energy = more likely
        produce_delta = torch.rand_like(energies) < prob
        
        n_delta = torch.sum(produce_delta).item()
        
        if n_delta == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros(0, device=self.device)
            )
        
        # Delta-ray positions
        delta_positions = positions[produce_delta]
        
        # Delta-ray energies (sample from Møller distribution)
        # Simplified: exponential distribution
        max_energy = energies[produce_delta] / 2.0  # Can't transfer more than half
        delta_energies = -torch.log(torch.rand(n_delta, device=self.device)) * 10.0
        delta_energies = torch.clamp(delta_energies, min=delta_ray_threshold, max=max_energy)
        
        return delta_positions, delta_energies
    
    def _calculate_beta(self, energies: torch.Tensor) -> torch.Tensor:
        """Calculate relativistic beta (v/c).
        
        Args:
            energies: Kinetic energies in keV [N]
            
        Returns:
            Beta values [N]
        """
        # β = sqrt(1 - (m₀c²/(E + m₀c²))²)
        total_energy = energies + self.electron_rest_mass_keV
        gamma = total_energy / self.electron_rest_mass_keV
        beta = torch.sqrt(1.0 - 1.0 / (gamma ** 2))
        
        return beta
    
    def _get_radiation_length(self, z: float, density: torch.Tensor) -> torch.Tensor:
        """Get radiation length for material.
        
        Args:
            z: Atomic number
            density: Density in g/cm³
            
        Returns:
            Radiation length in cm
        """
        # Approximate formula: X₀ ≈ 716.4 A / (Z(Z+1) ln(287/√Z)) g/cm²
        # For soft tissue (approximate)
        A = 7.4  # Average mass number
        X0_mass = RADIATION_LENGTH_CONSTANT * A / (
            z * (z + 1) * np.log(RADIATION_LENGTH_LOG_CONSTANT / np.sqrt(z))
        )
        
        # Convert to length: X₀ = X₀_mass / ρ
        X0_length = X0_mass / density
        
        return X0_length
    
    def _sample_bremsstrahlung_angle(
        self,
        electron_energies: torch.Tensor,
        photon_energies: torch.Tensor
    ) -> torch.Tensor:
        """Sample bremsstrahlung photon emission angle.
        
        Args:
            electron_energies: Electron energies [N]
            photon_energies: Photon energies [N]
            
        Returns:
            Cosine of emission angle [N]
        """
        # Bremsstrahlung is forward-peaked
        # Characteristic angle: θ ≈ m₀c²/E
        
        characteristic_angle = self.electron_rest_mass_keV / electron_energies
        
        # Sample from exponential distribution
        rand = torch.rand_like(electron_energies)
        theta = characteristic_angle * (-torch.log(rand + EPSILON))
        
        cos_theta = torch.cos(theta)
        
        return cos_theta
    
    def _rotate_direction(
        self,
        directions: torch.Tensor,
        cos_theta: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """Rotate direction vectors.
        
        Args:
            directions: Original directions [N, 3]
            cos_theta: Cosine of polar angle [N]
            phi: Azimuthal angle [N]
            
        Returns:
            Rotated directions [N, 3]
        """
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        # Simplified rotation
        new_directions = torch.zeros_like(directions)
        new_directions[:, 0] = sin_theta * cos_phi
        new_directions[:, 1] = sin_theta * sin_phi
        new_directions[:, 2] = cos_theta
        
        # Normalize
        norm = torch.sqrt(torch.sum(new_directions ** 2, dim=1, keepdim=True))
        new_directions = new_directions / (norm + EPSILON)
        
        return new_directions
