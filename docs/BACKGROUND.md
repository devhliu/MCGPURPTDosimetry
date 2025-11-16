# Background and Motivation

## Introduction

MCGPURPTDosimetry is a GPU-accelerated Monte Carlo simulation platform designed for internal dosimetry calculations in radiopharmaceutical therapy (RPT). This document provides the scientific and technical background for understanding the system's purpose, design decisions, and implementation approach.

## Medical Context

### Radiopharmaceutical Therapy (RPT)

Radiopharmaceutical therapy involves administering radioactive compounds that selectively accumulate in target tissues (typically tumors) to deliver therapeutic radiation doses. Common applications include:

- **Peptide Receptor Radionuclide Therapy (PRRT)**: Using Lu-177 or Y-90 labeled peptides for neuroendocrine tumors
- **Radioimmunotherapy**: Using antibodies labeled with therapeutic radionuclides
- **Thyroid Cancer Treatment**: Using I-131 for differentiated thyroid cancer
- **Targeted Alpha Therapy**: Using Ac-225 or At-211 for metastatic disease

### The Dosimetry Challenge

Accurate dose calculation in RPT is critical for:

1. **Treatment Planning**: Determining optimal administered activity
2. **Toxicity Prediction**: Avoiding damage to organs at risk (kidneys, bone marrow)
3. **Response Prediction**: Correlating dose with therapeutic outcome
4. **Regulatory Compliance**: Meeting safety requirements for clinical trials

Traditional dosimetry methods (MIRD formalism, S-values) make simplifying assumptions about uniform activity distribution and homogeneous tissue composition. These assumptions break down for:

- Heterogeneous activity distributions from SPECT/PET imaging
- Patient-specific anatomy and tissue composition
- Complex decay chains with multiple radiation types
- Non-uniform tissue density (bone, lung, contrast-enhanced regions)

### Monte Carlo Dosimetry

Monte Carlo simulation provides the gold standard for dose calculation by:

- Explicitly tracking individual particle interactions
- Accounting for tissue heterogeneity from CT imaging
- Handling complex radiation spectra (beta, gamma, alpha)
- Calculating dose distributions at voxel resolution

However, traditional Monte Carlo codes (Geant4, MCNP, PENELOPE) are computationally expensive, requiring hours to days for clinical cases.

## Technical Motivation

### The GPU Acceleration Opportunity

Modern GPUs offer:

- **Massive Parallelism**: Thousands of cores for simultaneous particle transport
- **High Memory Bandwidth**: Fast access to geometry and physics data
- **Mature Ecosystem**: PyTorch provides robust tensor operations and CUDA integration

By implementing Monte Carlo transport on GPU, we achieve:

- **10-20x speedup** over CPU implementations
- **Clinical feasibility**: Minutes instead of hours for patient-specific calculations
- **Interactive workflows**: Rapid iteration for treatment planning

### Design Philosophy

MCGPURPTDosimetry is built on three core principles:

1. **Production-Grade Physics**: Implement validated physics models comparable to established codes (±5-10% accuracy)
2. **GPU-First Architecture**: Design algorithms specifically for GPU parallelism, not just port CPU code
3. **Clinical Usability**: Provide simple APIs, flexible I/O, and comprehensive documentation

## Scientific Background

### Radiation Transport Physics

The system simulates four particle types with distinct transport characteristics:

#### Photons (Gamma Rays, X-rays)

**Interaction Mechanisms**:
- **Photoelectric Effect**: Dominant at low energies (<100 keV), important in high-Z materials
- **Compton Scattering**: Dominant at intermediate energies (100 keV - 10 MeV)
- **Pair Production**: Dominant at high energies (>10 MeV)
- **Rayleigh Scattering**: Elastic scattering, minor contribution

**Transport Characteristics**:
- Long mean free path (cm scale in tissue)
- Discrete interactions with stochastic outcomes
- Generate secondary electrons and photons

#### Electrons and Positrons

**Interaction Mechanisms**:
- **Ionization**: Continuous energy loss to atomic electrons
- **Bremsstrahlung**: Radiative energy loss producing photons
- **Multiple Coulomb Scattering**: Angular deflection from nuclear fields
- **Delta-Ray Production**: Energetic knock-on electrons

**Transport Characteristics**:
- Short range (mm scale in tissue)
- Thousands of interactions per particle
- Condensed history approach groups interactions into macrosteps

#### Alpha Particles

**Interaction Mechanisms**:
- **Ionization**: Extremely high linear energy transfer (LET)
- **Nuclear Reactions**: Rare, typically ignored

**Transport Characteristics**:
- Very short range (40-120 μm in tissue)
- Local energy deposition approximation (range << voxel size)
- Critical for targeted alpha therapy

### Beta Decay Spectra

Beta particles are emitted with a continuous energy spectrum from zero to a maximum endpoint energy. The spectrum shape follows Fermi theory:

```
N(E) ∝ p(E) × E × (E_max - E)² × F(Z, E)
```

where:
- `p(E)` is the electron momentum
- `E_max` is the endpoint energy
- `F(Z, E)` is the Fermi function (Coulomb correction)

Accurate sampling of beta spectra is essential for realistic dose distributions, as the energy distribution affects:
- Electron range and penetration depth
- Bremsstrahlung photon production
- Dose deposition patterns

### Decay Chains

Many therapeutic radionuclides undergo sequential decays through daughter products:

**Example: Ac-225 Decay Chain**
```
Ac-225 (α, 10 d) → Fr-221 (α, 4.8 min) → At-217 (α, 32 ms) → 
Bi-213 (α/β, 46 min) → Po-213 (α, 4 μs) / Tl-209 (β, 2.2 min) → 
Pb-209 (stable)
```

Each decay produces different radiation types with different energies, requiring:
- Tracking of daughter nuclide buildup
- Separate dose contributions from each nuclide
- Time-integrated activity calculations

## Computational Approach

### Condensed History Method

For charged particles (electrons, positrons), the condensed history approach groups thousands of microscopic interactions into macroscopic steps:

**Advantages**:
- Reduces computational cost by 1000x compared to detailed simulation
- Maintains accuracy for dose deposition (±10%)
- Enables GPU parallelization

**Implementation**:
- Step size limited by energy loss and geometry
- Continuous slowing down approximation (CSDA) for energy loss
- Multiple scattering for angular deflection
- Explicit simulation of hard interactions (bremsstrahlung, delta rays)

### Vectorized GPU Operations

Traditional Monte Carlo codes simulate particles sequentially. GPU implementation requires:

**Batch Processing**:
- Process thousands of particles simultaneously
- Vectorized operations on particle state tensors
- Coalesced memory access patterns

**Dynamic Stack Management**:
- Pre-allocated particle stacks on GPU memory
- Compact operations to remove terminated particles
- Secondary particle buffers for interaction products

**Atomic Operations**:
- Thread-safe dose accumulation
- Collision counters and statistics

## Validation and Accuracy

### Benchmarking Strategy

The system is validated against established Monte Carlo codes:

- **Geant4**: General-purpose particle transport (gold standard)
- **PENELOPE**: Specialized for low-energy photon/electron transport
- **MCNP**: Neutron and photon transport (nuclear industry standard)

### Accuracy Targets

- **Photon Dose**: ±5% compared to Geant4
- **Electron Dose**: ±10% compared to PENELOPE
- **Total Dose**: ±5-10% for mixed radiation fields

### Known Limitations

- Condensed history approximation introduces systematic errors for very low energies (<10 keV)
- Alpha local deposition assumes range << voxel size (valid for typical voxel sizes >1 mm)
- Simplified physics models (e.g., Gaussian vs Landau straggling) trade accuracy for speed

## Clinical Workflow Integration

### Input Requirements

1. **CT Image**: Hounsfield units for tissue composition and density
2. **Activity Map**: SPECT or PET image showing radiopharmaceutical distribution
3. **Radionuclide**: Specification of isotope and decay data
4. **Simulation Parameters**: Number of particles, energy cutoffs, etc.

### Output Products

1. **Dose Maps**: 3D dose distribution in Gy
2. **Uncertainty Maps**: Statistical uncertainty per voxel
3. **Dose-Volume Histograms**: For organs of interest
4. **Summary Statistics**: Mean dose, max dose, dose metrics

### Performance Requirements

For clinical adoption, dosimetry calculations must be:

- **Fast**: <5 minutes for typical patient cases
- **Accurate**: Within 10% of gold standard methods
- **Reliable**: Robust error handling and validation
- **Accessible**: Simple APIs and clear documentation

MCGPURPTDosimetry meets these requirements through GPU acceleration, validated physics, and user-friendly design.

## Future Directions

### Short-Term Enhancements

- Multi-timepoint imaging for time-dependent activity
- Automated organ segmentation integration
- Dose-volume histogram calculation
- Comprehensive test suite

### Long-Term Research

- Machine learning surrogate models for ultra-fast dose estimation
- Uncertainty quantification beyond statistical errors
- Biological dose modeling (RBE, LET effects)
- Treatment planning optimization

## References

### Monte Carlo Methods

1. Salvat F, et al. "PENELOPE-2018: A Code System for Monte Carlo Simulation of Electron and Photon Transport" (2019)
2. Agostinelli S, et al. "Geant4—a simulation toolkit" Nuclear Instruments and Methods (2003)
3. Kawrakow I. "Accurate condensed history Monte Carlo simulation of electron transport" Medical Physics (2000)

### Internal Dosimetry

4. Bolch WE, et al. "MIRD Pamphlet No. 21: A Generalized Schema for Radiopharmaceutical Dosimetry" Journal of Nuclear Medicine (2009)
5. Ljungberg M, et al. "MIRD Pamphlet No. 26: Joint EANM/MIRD Guidelines for Quantitative 177Lu SPECT" Journal of Nuclear Medicine (2016)

### GPU Computing

6. Jia X, et al. "GPU-based fast Monte Carlo simulation for radiotherapy dose calculation" Physics in Medicine & Biology (2011)
7. Hissoiny S, et al. "GPUMCD: A new GPU-oriented Monte Carlo dose calculation platform" Medical Physics (2011)

---

**Next**: See [REQUIREMENTS.md](REQUIREMENTS.md) for system requirements and dependencies.
