"""Detailed photon interaction physics."""

import torch
import numpy as np
from typing import Tuple, Optional

from ..utils.logging import get_logger
from .constants import (
    ELECTRON_REST_MASS_KEV,
    EPSILON,
    MAX_REJECTION_ITERATIONS,
    BINDING_ENERGY_CONSTANT,
    FLUORESCENCE_YIELD_CONSTANT,
    CHARACTERISTIC_ENERGY_CONSTANT
)


logger = get_logger()


class PhotonPhysics:
    """Detailed photon interaction physics.
    
    Implements accurate photon transport including:
    - Photoelectric effect with fluorescence
    - Compton scattering (Klein-Nishina)
    - Pair production
    - Rayleigh (coherent) scattering
    
    Attributes:
        device: Computation device
        electron_rest_mass_keV: Electron rest mass energy (511 keV)
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize PhotonPhysics.
        
        Args:
            device: Computation device
        """
        self.device = device
        self.electron_rest_mass_keV = ELECTRON_REST_MASS_KEV
        logger.debug("PhotonPhysics initialized")
    
    def sample_interaction_type(
        self,
        energies: torch.Tensor,
        photoelectric_xs: torch.Tensor,
        compton_xs: torch.Tensor,
        pair_xs: torch.Tensor,
        rayleigh_xs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample interaction type based on cross-sections.
        
        Args:
            energies: Photon energies [N]
            photoelectric_xs: Photoelectric cross-sections [N]
            compton_xs: Compton cross-sections [N]
            pair_xs: Pair production cross-sections [N]
            rayleigh_xs: Optional Rayleigh cross-sections [N]
            
        Returns:
            Interaction types [N]: 0=photoelectric, 1=Compton, 2=pair, 3=Rayleigh
        """
        # Calculate total cross-section
        total_xs = photoelectric_xs + compton_xs + pair_xs
        if rayleigh_xs is not None:
            total_xs += rayleigh_xs
        
        # Sample random numbers
        rand = torch.rand(len(energies), device=self.device)
        
        # Cumulative probabilities
        prob_photo = photoelectric_xs / total_xs
        prob_compton = (photoelectric_xs + compton_xs) / total_xs
        prob_pair = (photoelectric_xs + compton_xs + pair_xs) / total_xs
        
        # Determine interaction type
        interaction_type = torch.zeros(len(energies), dtype=torch.int32, device=self.device)
        interaction_type[rand < prob_photo] = 0  # Photoelectric
        interaction_type[(rand >= prob_photo) & (rand < prob_compton)] = 1  # Compton
        interaction_type[(rand >= prob_compton) & (rand < prob_pair)] = 2  # Pair production
        if rayleigh_xs is not None:
            interaction_type[rand >= prob_pair] = 3  # Rayleigh
        
        return interaction_type
    
    def sample_free_path(
        self,
        energies: torch.Tensor,
        total_xs: torch.Tensor,
        density: torch.Tensor
    ) -> torch.Tensor:
        """Sample free path to next interaction.
        
        Args:
            energies: Photon energies [N]
            total_xs: Total cross-sections in cm²/g [N]
            density: Material density in g/cm³ [N]
            
        Returns:
            Free path distances in cm [N]
        """
        # Mean free path: λ = 1 / (μ * ρ) where μ is mass attenuation coefficient
        mean_free_path = 1.0 / (total_xs * density + EPSILON)
        
        # Sample exponential distribution
        rand = torch.rand(len(energies), device=self.device)
        free_path = -mean_free_path * torch.log(rand + EPSILON)
        
        return free_path
    
    def photoelectric_interaction(
        self,
        energies: torch.Tensor,
        positions: torch.Tensor,
        material_z: float = 7.4  # Effective Z for soft tissue
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate photoelectric effect with fluorescence.
        
        Args:
            energies: Photon energies [N]
            positions: Photon positions [N, 3]
            material_z: Effective atomic number
            
        Returns:
            Tuple of (electron_positions, electron_energies, 
                     fluorescence_positions, fluorescence_energies)
        """
        n_photons = len(energies)
        
        # Photoelectron energy (photon energy minus binding energy)
        # Simplified: assume K-shell binding energy
        binding_energy = self._get_binding_energy(material_z)
        electron_energies = torch.clamp(energies - binding_energy, min=0.0)
        
        # Photoelectron direction (simplified: isotropic)
        theta = torch.acos(2 * torch.rand(n_photons, device=self.device) - 1)
        phi = 2 * np.pi * torch.rand(n_photons, device=self.device)
        
        # Fluorescence yield (probability of characteristic X-ray emission)
        fluorescence_yield = self._get_fluorescence_yield(material_z)
        
        # Sample which interactions produce fluorescence
        produce_fluorescence = torch.rand(n_photons, device=self.device) < fluorescence_yield
        
        # Fluorescence photon energies (characteristic X-rays)
        fluorescence_energies = torch.zeros(n_photons, device=self.device)
        fluorescence_energies[produce_fluorescence] = self._get_characteristic_energy(material_z)
        
        return (
            positions,  # Electron positions (same as interaction site)
            electron_energies,
            positions,  # Fluorescence positions (same location)
            fluorescence_energies
        )
    
    def compton_scattering(
        self,
        energies: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate Compton scattering using Klein-Nishina formula.
        
        Args:
            energies: Photon energies in keV [N]
            directions: Photon directions [N, 3]
            
        Returns:
            Tuple of (scattered_energies, scattered_directions,
                     electron_energies, electron_directions)
        """
        n_photons = len(energies)
        
        # Photon energy in units of electron rest mass
        alpha = energies / self.electron_rest_mass_keV
        
        # Sample scattering angle using Klein-Nishina distribution
        cos_theta = self._sample_klein_nishina_angle(alpha)
        
        # Calculate scattered photon energy
        # E' = E / (1 + α(1 - cos θ))
        scattered_energies = energies / (1.0 + alpha * (1.0 - cos_theta))
        
        # Recoil electron energy
        electron_energies = energies - scattered_energies
        
        # Sample azimuthal angle uniformly
        phi = 2 * np.pi * torch.rand(n_photons, device=self.device)
        
        # Calculate scattered photon direction
        scattered_directions = self._rotate_direction(
            directions, cos_theta, phi
        )
        
        # Electron direction (momentum conservation)
        # Simplified: perpendicular to photon scattering plane
        electron_directions = self._calculate_electron_direction(
            directions, scattered_directions, energies, electron_energies
        )
        
        return (
            scattered_energies,
            scattered_directions,
            electron_energies,
            electron_directions
        )
    
    def pair_production(
        self,
        energies: torch.Tensor,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate pair production.
        
        Args:
            energies: Photon energies [N]
            positions: Photon positions [N, 3]
            
        Returns:
            Tuple of (electron_positions, electron_energies,
                     positron_positions, positron_energies)
        """
        n_photons = len(energies)
        
        # Available energy for e+e- pair (photon energy - 2*m_e*c²)
        available_energy = energies - 2 * self.electron_rest_mass_keV
        
        # Share energy between electron and positron
        # Simplified: equal sharing
        electron_energies = available_energy / 2.0
        positron_energies = available_energy / 2.0
        
        # Directions (simplified: isotropic)
        theta_e = torch.acos(2 * torch.rand(n_photons, device=self.device) - 1)
        phi_e = 2 * np.pi * torch.rand(n_photons, device=self.device)
        
        theta_p = torch.acos(2 * torch.rand(n_photons, device=self.device) - 1)
        phi_p = 2 * np.pi * torch.rand(n_photons, device=self.device)
        
        return (
            positions,  # Electron positions
            electron_energies,
            positions,  # Positron positions (same location)
            positron_energies
        )
    
    def rayleigh_scattering(
        self,
        directions: torch.Tensor
    ) -> torch.Tensor:
        """Simulate Rayleigh (coherent) scattering.
        
        Args:
            directions: Photon directions [N, 3]
            
        Returns:
            Scattered directions [N, 3]
        """
        n_photons = len(directions)
        
        # Rayleigh scattering: elastic, forward-peaked
        # Sample scattering angle from form factor
        cos_theta = self._sample_rayleigh_angle(n_photons)
        phi = 2 * np.pi * torch.rand(n_photons, device=self.device)
        
        # Rotate direction
        scattered_directions = self._rotate_direction(directions, cos_theta, phi)
        
        return scattered_directions
    
    def _sample_klein_nishina_angle(self, alpha: torch.Tensor) -> torch.Tensor:
        """Sample scattering angle from Klein-Nishina distribution.
        
        Uses vectorized rejection sampling for GPU efficiency.
        
        Args:
            alpha: Photon energy in units of electron rest mass [N]
            
        Returns:
            Cosine of scattering angle [N]
        """
        n = len(alpha)
        cos_theta = torch.zeros(n, device=self.device)
        remaining_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        
        # Vectorized rejection sampling
        for iteration in range(MAX_REJECTION_ITERATIONS):
            n_remaining = remaining_mask.sum().item()
            if n_remaining == 0:
                break
            
            # Sample cos_theta uniformly for remaining particles
            cos_theta_trial = 2 * torch.rand(n_remaining, device=self.device) - 1
            
            # Klein-Nishina differential cross-section
            alpha_remaining = alpha[remaining_mask]
            denominator = 1.0 + alpha_remaining * (1.0 - cos_theta_trial)
            
            kn_value = (1.0 / (denominator ** 2)) * (
                1.0 + cos_theta_trial ** 2 + 
                (alpha_remaining ** 2 * (1.0 - cos_theta_trial) ** 2) / denominator
            )
            
            # Rejection test (vectorized)
            accept_prob = kn_value / 2.0
            accept_mask = torch.rand(n_remaining, device=self.device) < accept_prob
            
            # Update accepted values
            accepted_indices = torch.where(remaining_mask)[0][accept_mask]
            cos_theta[accepted_indices] = cos_theta_trial[accept_mask]
            remaining_mask[accepted_indices] = False
        
        # Fallback for any remaining unaccepted particles
        if remaining_mask.any():
            n_fallback = remaining_mask.sum().item()
            logger.warning(f"{n_fallback} particles used fallback Klein-Nishina sampling")
            cos_theta[remaining_mask] = 2 * torch.rand(n_fallback, device=self.device) - 1
        
        return cos_theta
    
    def _sample_rayleigh_angle(self, n: int) -> torch.Tensor:
        """Sample Rayleigh scattering angle.
        
        Args:
            n: Number of samples
            
        Returns:
            Cosine of scattering angle [n]
        """
        # Rayleigh scattering is forward-peaked
        # Simplified: use exponential distribution
        rand = torch.rand(n, device=self.device)
        cos_theta = 1.0 - 2.0 * rand ** 2
        
        return cos_theta
    
    def _rotate_direction(
        self,
        directions: torch.Tensor,
        cos_theta: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """Rotate direction vector by angles.
        
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
        
        # Rotation matrix application (simplified)
        # This is a simplified rotation; full implementation would use proper 3D rotation
        new_directions = torch.zeros_like(directions)
        new_directions[:, 0] = sin_theta * cos_phi
        new_directions[:, 1] = sin_theta * sin_phi
        new_directions[:, 2] = cos_theta
        
        # Normalize
        norm = torch.sqrt(torch.sum(new_directions ** 2, dim=1, keepdim=True))
        new_directions = new_directions / (norm + 1e-30)
        
        return new_directions
    
    def _calculate_electron_direction(
        self,
        photon_dir: torch.Tensor,
        scattered_dir: torch.Tensor,
        photon_energy: torch.Tensor,
        electron_energy: torch.Tensor
    ) -> torch.Tensor:
        """Calculate recoil electron direction from momentum conservation.
        
        Args:
            photon_dir: Initial photon direction [N, 3]
            scattered_dir: Scattered photon direction [N, 3]
            photon_energy: Initial photon energy [N]
            electron_energy: Recoil electron energy [N]
            
        Returns:
            Electron direction [N, 3]
        """
        # Momentum conservation (simplified)
        # p_e = p_γ - p_γ'
        
        # Photon momentum is proportional to energy
        p_initial = photon_dir * photon_energy.unsqueeze(1)
        p_scattered = scattered_dir * (photon_energy - electron_energy).unsqueeze(1)
        
        p_electron = p_initial - p_scattered
        
        # Normalize
        norm = torch.sqrt(torch.sum(p_electron ** 2, dim=1, keepdim=True))
        electron_dir = p_electron / (norm + 1e-30)
        
        return electron_dir
    
    def _get_binding_energy(self, z: float) -> float:
        """Get K-shell binding energy for element.
        
        Args:
            z: Atomic number
            
        Returns:
            Binding energy in keV
        """
        # Simplified formula: E_K ≈ 13.6 * Z² / n² eV
        # For K-shell (n=1)
        return BINDING_ENERGY_CONSTANT * z ** 2  # Convert eV to keV
    
    def _get_fluorescence_yield(self, z: float) -> float:
        """Get fluorescence yield for element.
        
        Args:
            z: Atomic number
            
        Returns:
            Fluorescence yield (0-1)
        """
        # Empirical formula for K-shell fluorescence yield
        # ω_K ≈ Z⁴ / (A + Z⁴) where A ≈ 10⁶
        return (z ** 4) / (FLUORESCENCE_YIELD_CONSTANT + z ** 4)
    
    def _get_characteristic_energy(self, z: float) -> float:
        """Get characteristic X-ray energy for element.
        
        Args:
            z: Atomic number
            
        Returns:
            Characteristic energy in keV
        """
        # K-alpha energy (simplified)
        # E_Kα ≈ 10.2 * (Z - 1)² eV
        return CHARACTERISTIC_ENERGY_CONSTANT * (z - 1) ** 2  # Convert eV to keV
