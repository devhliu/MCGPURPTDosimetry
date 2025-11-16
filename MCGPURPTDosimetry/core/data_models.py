"""Core data models for the dosimetry system."""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import torch
import numpy as np


@dataclass
class GeometryData:
    """3D geometry data for Monte Carlo simulation.
    
    Attributes:
        material_map: Tensor of material IDs [X, Y, Z]
        density_map: Tensor of densities in g/cm³ [X, Y, Z]
        voxel_size: Voxel dimensions in mm (dx, dy, dz)
        dimensions: Grid dimensions (nx, ny, nz)
        affine_matrix: 4x4 spatial transform matrix
    """
    material_map: torch.Tensor
    density_map: torch.Tensor
    voxel_size: Tuple[float, float, float]
    dimensions: Tuple[int, int, int]
    affine_matrix: np.ndarray


@dataclass
class ParticleStack:
    """Dynamic stack for particle transport on GPU.
    
    Attributes:
        positions: Particle positions in voxel coordinates [N, 3]
        directions: Unit direction vectors [N, 3]
        energies: Particle energies in keV [N]
        weights: Statistical weights [N]
        active_mask: Boolean mask for active particles [N]
        num_active: Count of currently active particles
        capacity: Maximum stack capacity
    """
    positions: torch.Tensor
    directions: torch.Tensor
    energies: torch.Tensor
    weights: torch.Tensor
    active_mask: torch.Tensor
    num_active: int
    capacity: int
    
    @classmethod
    def create_empty(cls, capacity: int, device: str = 'cuda') -> 'ParticleStack':
        """Create an empty particle stack with pre-allocated memory."""
        return cls(
            positions=torch.zeros((capacity, 3), dtype=torch.float32, device=device),
            directions=torch.zeros((capacity, 3), dtype=torch.float32, device=device),
            energies=torch.zeros(capacity, dtype=torch.float32, device=device),
            weights=torch.ones(capacity, dtype=torch.float32, device=device),
            active_mask=torch.zeros(capacity, dtype=torch.bool, device=device),
            num_active=0,
            capacity=capacity
        )
    
    def get_active(self) -> 'ParticleStack':
        """Return a view of only active particles."""
        if self.num_active == 0:
            return ParticleStack.create_empty(0, device=self.positions.device)
        
        active_indices = torch.where(self.active_mask)[0]
        return ParticleStack(
            positions=self.positions[active_indices],
            directions=self.directions[active_indices],
            energies=self.energies[active_indices],
            weights=self.weights[active_indices],
            active_mask=self.active_mask[active_indices],
            num_active=self.num_active,
            capacity=self.num_active
        )
    
    def add_particles(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        energies: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> None:
        """Add new particles to the stack."""
        n_new = len(energies)
        if n_new == 0:
            return
        
        if self.num_active + n_new > self.capacity:
            raise RuntimeError(
                f"Stack overflow: trying to add {n_new} particles to stack "
                f"with {self.capacity - self.num_active} free slots"
            )
        
        start_idx = self.num_active
        end_idx = self.num_active + n_new
        
        self.positions[start_idx:end_idx] = positions
        self.directions[start_idx:end_idx] = directions
        self.energies[start_idx:end_idx] = energies
        if weights is not None:
            self.weights[start_idx:end_idx] = weights
        else:
            self.weights[start_idx:end_idx] = 1.0
        self.active_mask[start_idx:end_idx] = True
        self.num_active += n_new
    
    def compact(self) -> None:
        """Remove inactive particles to maintain efficiency."""
        if self.num_active == 0:
            return
        
        active_indices = torch.where(self.active_mask)[0]
        n_active = len(active_indices)
        
        if n_active == self.num_active:
            return  # Already compact
        
        # Move active particles to front
        self.positions[:n_active] = self.positions[active_indices]
        self.directions[:n_active] = self.directions[active_indices]
        self.energies[:n_active] = self.energies[active_indices]
        self.weights[:n_active] = self.weights[active_indices]
        self.active_mask[:n_active] = True
        self.active_mask[n_active:] = False
        
        self.num_active = n_active


@dataclass
class SecondaryParticleBuffer:
    """Thread-safe buffer for secondary particles generated during interactions.
    
    Attributes:
        photon_positions: Secondary photon positions [M, 3]
        photon_directions: Secondary photon directions [M, 3]
        photon_energies: Secondary photon energies [M]
        photon_weights: Secondary photon weights [M]
        photon_count: Number of photons in buffer
        electron_positions: Secondary electron positions [K, 3]
        electron_directions: Secondary electron directions [K, 3]
        electron_energies: Secondary electron energies [K]
        electron_weights: Secondary electron weights [K]
        electron_count: Number of electrons in buffer
        max_capacity: Maximum buffer capacity
    """
    photon_positions: torch.Tensor
    photon_directions: torch.Tensor
    photon_energies: torch.Tensor
    photon_weights: torch.Tensor
    photon_count: torch.Tensor  # Atomic counter
    
    electron_positions: torch.Tensor
    electron_directions: torch.Tensor
    electron_energies: torch.Tensor
    electron_weights: torch.Tensor
    electron_count: torch.Tensor  # Atomic counter
    
    max_capacity: int
    
    @classmethod
    def create_empty(cls, max_capacity: int, device: str = 'cuda') -> 'SecondaryParticleBuffer':
        """Create an empty secondary particle buffer."""
        return cls(
            photon_positions=torch.zeros((max_capacity, 3), dtype=torch.float32, device=device),
            photon_directions=torch.zeros((max_capacity, 3), dtype=torch.float32, device=device),
            photon_energies=torch.zeros(max_capacity, dtype=torch.float32, device=device),
            photon_weights=torch.ones(max_capacity, dtype=torch.float32, device=device),
            photon_count=torch.zeros(1, dtype=torch.int32, device=device),
            electron_positions=torch.zeros((max_capacity, 3), dtype=torch.float32, device=device),
            electron_directions=torch.zeros((max_capacity, 3), dtype=torch.float32, device=device),
            electron_energies=torch.zeros(max_capacity, dtype=torch.float32, device=device),
            electron_weights=torch.ones(max_capacity, dtype=torch.float32, device=device),
            electron_count=torch.zeros(1, dtype=torch.int32, device=device),
            max_capacity=max_capacity
        )
    
    def clear(self) -> None:
        """Reset buffer for next iteration."""
        self.photon_count.zero_()
        self.electron_count.zero_()
    
    def flush_to_stacks(
        self,
        photon_stack: ParticleStack,
        electron_stack: ParticleStack
    ) -> None:
        """Transfer accumulated secondaries to main transport stacks."""
        n_photons = self.photon_count.item()
        n_electrons = self.electron_count.item()
        
        if n_photons > 0:
            photon_stack.add_particles(
                self.photon_positions[:n_photons],
                self.photon_directions[:n_photons],
                self.photon_energies[:n_photons],
                self.photon_weights[:n_photons]
            )
        
        if n_electrons > 0:
            electron_stack.add_particles(
                self.electron_positions[:n_electrons],
                self.electron_directions[:n_electrons],
                self.electron_energies[:n_electrons],
                self.electron_weights[:n_electrons]
            )


@dataclass
class EmissionData:
    """Data for a single radiation emission.
    
    Attributes:
        particle_type: Type of particle ('alpha', 'beta_minus', 'beta_plus', 'gamma', 'electron', 'positron')
        energy_keV: Emission energy in keV (mean energy for beta decay)
        intensity: Emissions per decay
        max_energy_keV: Maximum energy for beta spectrum (None for discrete emissions)
    """
    particle_type: str
    energy_keV: float
    intensity: float
    max_energy_keV: Optional[float] = None


@dataclass
class DecayMode:
    """Data for a nuclear decay mode.
    
    Attributes:
        mode_type: Type of decay ('beta_minus', 'beta_plus', 'alpha', etc.)
        branching_ratio: Probability of this decay mode
        emissions: List of radiation emissions from this decay
    """
    mode_type: str
    branching_ratio: float
    emissions: List[EmissionData]


@dataclass
class NuclideData:
    """Complete decay data for a radionuclide.
    
    Attributes:
        name: Nuclide name (e.g., 'Lu-177')
        half_life_seconds: Physical half-life in seconds
        decay_modes: Dictionary of decay modes
        daughters: List of daughter nuclide names
    """
    name: str
    half_life_seconds: float
    decay_modes: Dict[str, DecayMode]
    daughters: List[str] = field(default_factory=list)


@dataclass
class CrossSectionData:
    """Photon interaction cross-section data.
    
    Attributes:
        energy_grid: Energy points in keV [N]
        photoelectric: Photoelectric cross-section in cm²/g [N]
        compton: Compton cross-section in cm²/g [N]
        pair_production: Pair production cross-section in cm²/g [N]
        total: Total cross-section in cm²/g [N]
    """
    energy_grid: torch.Tensor
    photoelectric: torch.Tensor
    compton: torch.Tensor
    pair_production: torch.Tensor
    total: torch.Tensor
    
    def interpolate(self, energy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Interpolate cross-sections at given energies.
        
        Args:
            energy: Energies to interpolate at [M]
            
        Returns:
            Dictionary with interpolated cross-sections
        """
        # Log-log interpolation for cross-sections
        log_energy = torch.log(energy)
        log_energy_grid = torch.log(self.energy_grid)
        
        result = {}
        for name, data in [
            ('photoelectric', self.photoelectric),
            ('compton', self.compton),
            ('pair_production', self.pair_production),
            ('total', self.total)
        ]:
            log_data = torch.log(data + 1e-30)  # Avoid log(0)
            
            # Manual 1D linear interpolation using torch operations
            # Find indices for interpolation
            log_interp = self._torch_interp_1d(log_energy, log_energy_grid, log_data)
            result[name] = torch.exp(log_interp)
        
        return result
    
    @staticmethod
    def _torch_interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """1D linear interpolation using pure torch operations.
        
        Args:
            x: Query points [M]
            xp: Data point x-coordinates [N] (must be sorted)
            fp: Data point y-coordinates [N]
            
        Returns:
            Interpolated values [M]
        """
        # Ensure x is 1D
        x_shape = x.shape
        x = x.flatten()
        
        # Find indices where x would be inserted to maintain sorted order
        # searchsorted returns indices such that xp[i-1] <= x < xp[i]
        indices = torch.searchsorted(xp, x, right=False)
        
        # Clamp indices to valid range
        indices = torch.clamp(indices, 1, len(xp) - 1)
        
        # Get surrounding points
        x0 = xp[indices - 1]
        x1 = xp[indices]
        y0 = fp[indices - 1]
        y1 = fp[indices]
        
        # Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        slope = (y1 - y0) / (x1 - x0 + 1e-30)  # Avoid division by zero
        result = y0 + (x - x0) * slope
        
        # Reshape to original shape
        return result.reshape(x_shape)


@dataclass
class StoppingPowerData:
    """Electron stopping power data.
    
    Attributes:
        energy_grid: Energy points in keV [N]
        collisional: Collisional stopping power in MeV cm²/g [N]
        radiative: Radiative stopping power in MeV cm²/g [N]
        total: Total stopping power in MeV cm²/g [N]
        density_effect: Density effect correction [N]
    """
    energy_grid: torch.Tensor
    collisional: torch.Tensor
    radiative: torch.Tensor
    total: torch.Tensor
    density_effect: torch.Tensor
    
    def interpolate(self, energy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Interpolate stopping powers at given energies.
        
        Args:
            energy: Energies to interpolate at [M]
            
        Returns:
            Dictionary with interpolated stopping powers
        """
        # Log-log interpolation
        log_energy = torch.log(energy)
        log_energy_grid = torch.log(self.energy_grid)
        
        result = {}
        for name, data in [
            ('collisional', self.collisional),
            ('radiative', self.radiative),
            ('total', self.total),
            ('density_effect', self.density_effect)
        ]:
            log_data = torch.log(data + 1e-30)
            # Use the same torch-based interpolation method
            log_interp = CrossSectionData._torch_interp_1d(log_energy, log_energy_grid, log_data)
            result[name] = torch.exp(log_interp)
        
        return result


@dataclass
class MaterialProperties:
    """Physical properties of a material.
    
    Attributes:
        material_id: Unique integer identifier
        name: Material name
        density: Density in g/cm³
        composition: Elemental composition {element: mass_fraction}
        photon_cross_sections: Photon interaction data
        electron_stopping_powers: Electron stopping power data
    """
    material_id: int
    name: str
    density: float
    composition: Dict[str, float]
    photon_cross_sections: Optional[CrossSectionData] = None
    electron_stopping_powers: Optional[StoppingPowerData] = None
