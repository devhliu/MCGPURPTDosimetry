"""Beta spectrum sampling using the Alias method for O(1) performance."""

import torch
import numpy as np
from typing import Tuple, Optional

from ..utils.logging import get_logger


logger = get_logger()


class BetaSpectrumSampler:
    """Efficient beta spectrum sampling using the Alias method.
    
    The Alias method provides O(1) sampling from arbitrary discrete distributions
    by preprocessing the distribution into lookup tables. This is ideal for beta
    decay where we need to sample millions of energies from the same spectrum.
    
    Attributes:
        max_energy_keV: Maximum beta energy in keV
        n_bins: Number of energy bins
        energy_grid: Energy bin centers [n_bins]
        prob_table: Probability table for Alias method [n_bins]
        alias_table: Alias indices for Alias method [n_bins]
        device: Computation device
    """
    
    def __init__(
        self,
        max_energy_keV: float,
        n_bins: int = 1000,
        device: str = 'cuda'
    ):
        """Initialize BetaSpectrumSampler.
        
        Args:
            max_energy_keV: Maximum beta energy in keV
            n_bins: Number of energy bins for discretization
            device: Computation device ('cuda' or 'cpu')
        """
        self.max_energy_keV = max_energy_keV
        self.n_bins = n_bins
        self.device = device
        
        # Create energy grid
        self.energy_grid = torch.linspace(
            0, max_energy_keV, n_bins, device=device
        )
        
        # Calculate beta spectrum
        spectrum = self._calculate_beta_spectrum(self.energy_grid, max_energy_keV)
        
        # Build Alias tables
        self.prob_table, self.alias_table = self._build_alias_tables(spectrum)
        
        logger.debug(
            f"BetaSpectrumSampler initialized: "
            f"max_energy={max_energy_keV:.1f} keV, n_bins={n_bins}"
        )
    
    def _calculate_beta_spectrum(
        self,
        energies: torch.Tensor,
        max_energy: float
    ) -> torch.Tensor:
        """Calculate beta spectrum using Fermi theory.
        
        The beta spectrum follows the Fermi distribution:
        N(E) ∝ p * E * (E_max - E)² * F(Z, E)
        
        where:
        - p = sqrt(E² + 2*E*m_e*c²) is the momentum
        - E is the kinetic energy
        - E_max is the maximum energy
        - F(Z, E) is the Fermi function (approximated as 1 for simplicity)
        
        Args:
            energies: Energy grid in keV [n_bins]
            max_energy: Maximum beta energy in keV
            
        Returns:
            Normalized spectrum [n_bins]
        """
        # Electron rest mass energy in keV
        m_e_c2 = 511.0
        
        # Avoid division by zero at E=0
        E = energies + 1e-6
        
        # Calculate momentum: p = sqrt(E² + 2*E*m_e*c²)
        momentum = torch.sqrt(E**2 + 2*E*m_e_c2)
        
        # Beta spectrum: N(E) ∝ p * E * (E_max - E)²
        spectrum = momentum * E * (max_energy - E)**2
        
        # Set negative values to zero (beyond max energy)
        spectrum[energies > max_energy] = 0
        spectrum[spectrum < 0] = 0
        
        # Normalize
        spectrum = spectrum / (torch.sum(spectrum) + 1e-30)
        
        return spectrum
    
    def _build_alias_tables(
        self,
        probabilities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build Alias method lookup tables.
        
        The Alias method works by creating two tables:
        1. prob_table: Probability of using the primary bin
        2. alias_table: Index of the alias bin to use if not primary
        
        This allows O(1) sampling by:
        1. Randomly select a bin i
        2. Generate random u ~ U(0,1)
        3. If u < prob_table[i], return bin i
        4. Otherwise, return alias_table[i]
        
        Args:
            probabilities: Probability distribution [n_bins]
            
        Returns:
            Tuple of (prob_table, alias_table)
        """
        n = len(probabilities)
        
        # Scale probabilities to sum to n
        prob_scaled = probabilities * n
        
        # Initialize tables
        prob_table = torch.zeros(n, device=self.device)
        alias_table = torch.arange(n, device=self.device, dtype=torch.long)
        
        # Separate into small and large bins
        small = []
        large = []
        
        for i in range(n):
            if prob_scaled[i] < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Build alias table
        while small and large:
            s = small.pop()
            l = large.pop()
            
            prob_table[s] = prob_scaled[s]
            alias_table[s] = l
            
            prob_scaled[l] = prob_scaled[l] - (1.0 - prob_scaled[s])
            
            if prob_scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        
        # Remaining bins have probability 1
        while large:
            l = large.pop()
            prob_table[l] = 1.0
        
        while small:
            s = small.pop()
            prob_table[s] = 1.0
        
        return prob_table, alias_table
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample beta energies using the Alias method.
        
        This is an O(1) per sample operation after preprocessing.
        
        Args:
            n_samples: Number of energy samples to generate
            
        Returns:
            Sampled energies in keV [n_samples]
        """
        # Step 1: Randomly select bins
        bin_indices = torch.randint(
            0, self.n_bins, (n_samples,), device=self.device
        )
        
        # Step 2: Generate uniform random numbers
        u = torch.rand(n_samples, device=self.device)
        
        # Step 3: Use Alias method
        # If u < prob_table[i], use bin i; otherwise use alias_table[i]
        use_primary = u < self.prob_table[bin_indices]
        final_bins = torch.where(
            use_primary,
            bin_indices,
            self.alias_table[bin_indices]
        )
        
        # Step 4: Get energies from bins (with small random offset within bin)
        bin_width = self.max_energy_keV / self.n_bins
        energies = self.energy_grid[final_bins]
        
        # Add random offset within bin for smoother distribution
        offsets = (torch.rand(n_samples, device=self.device) - 0.5) * bin_width
        energies = energies + offsets
        
        # Clamp to valid range
        energies = torch.clamp(energies, min=0, max=self.max_energy_keV)
        
        return energies
    
    def get_mean_energy(self) -> float:
        """Get mean energy of the beta spectrum.
        
        Returns:
            Mean energy in keV
        """
        # Calculate spectrum
        spectrum = self._calculate_beta_spectrum(self.energy_grid, self.max_energy_keV)
        
        # Calculate mean
        mean_energy = torch.sum(self.energy_grid * spectrum).item()
        
        return mean_energy


class BetaSpectrumCache:
    """Cache of beta spectrum samplers for different nuclides.
    
    Maintains pre-computed Alias tables for all beta-emitting nuclides
    to enable fast sampling during simulation.
    
    Attributes:
        samplers: Dictionary mapping nuclide names to BetaSpectrumSampler instances
        device: Computation device
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize BetaSpectrumCache.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
        """
        self.samplers = {}
        self.device = device
        logger.debug("BetaSpectrumCache initialized")
    
    def add_nuclide(
        self,
        nuclide: str,
        max_energy_keV: float,
        n_bins: int = 1000
    ) -> None:
        """Add a beta spectrum sampler for a nuclide.
        
        Args:
            nuclide: Nuclide name
            max_energy_keV: Maximum beta energy in keV
            n_bins: Number of energy bins
        """
        if nuclide in self.samplers:
            logger.debug(f"Beta spectrum for {nuclide} already cached")
            return
        
        sampler = BetaSpectrumSampler(max_energy_keV, n_bins, self.device)
        self.samplers[nuclide] = sampler
        
        logger.info(
            f"Added beta spectrum for {nuclide}: "
            f"max_energy={max_energy_keV:.1f} keV, "
            f"mean_energy={sampler.get_mean_energy():.1f} keV"
        )
    
    def sample(self, nuclide: str, n_samples: int) -> Optional[torch.Tensor]:
        """Sample beta energies for a nuclide.
        
        Args:
            nuclide: Nuclide name
            n_samples: Number of samples
            
        Returns:
            Sampled energies in keV, or None if nuclide not in cache
        """
        if nuclide not in self.samplers:
            logger.warning(f"Beta spectrum for {nuclide} not in cache")
            return None
        
        return self.samplers[nuclide].sample(n_samples)
    
    def has_nuclide(self, nuclide: str) -> bool:
        """Check if nuclide is in cache.
        
        Args:
            nuclide: Nuclide name
            
        Returns:
            True if nuclide is cached
        """
        return nuclide in self.samplers
    
    def preload_from_decay_database(self, decay_db: 'DecayDatabase') -> None:
        """Preload beta spectra for all nuclides in decay database.
        
        Args:
            decay_db: Decay database instance
        """
        logger.info("Preloading beta spectra from decay database...")
        
        for nuclide_name in decay_db.list_nuclides():
            nuclide_data = decay_db.get_nuclide(nuclide_name)
            
            # Check for beta emissions
            for mode in nuclide_data.decay_modes.values():
                for emission in mode.emissions:
                    if emission.particle_type in ['beta_minus', 'beta_plus']:
                        # Beta decay has max_energy_keV field
                        if emission.max_energy_keV is not None:
                            self.add_nuclide(
                                nuclide_name,
                                emission.max_energy_keV
                            )
                            break  # Only need one beta spectrum per nuclide
        
        logger.info(f"Preloaded {len(self.samplers)} beta spectra")
