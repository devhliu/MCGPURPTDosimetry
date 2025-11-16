"""
GPU-Accelerated Internal Dosimetry Monte Carlo Calculation System

A PyTorch-based platform for calculating radiation dose distributions from 
therapeutic radiopharmaceuticals imaged via SPECT/PET.
"""

__version__ = "0.1.0"

from .core.dosimetry_simulator import DosimetrySimulator
from .utils.config import SimulationConfig

__all__ = ['DosimetrySimulator', 'SimulationConfig']
