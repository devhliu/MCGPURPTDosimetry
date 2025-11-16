# System Architecture and Design

## Overview

This document describes the software architecture, design patterns, and implementation strategies used in MCGPURPTDosimetry. Understanding the architecture is essential for developers who want to extend, modify, or contribute to the codebase.

## Architectural Principles

### 1. Separation of Concerns

The system is organized into distinct modules with clear responsibilities:

- **Core**: Simulation orchestration and data flow
- **Physics**: Particle transport and interaction models
- **Utils**: Configuration, logging, validation
- **Data**: Physics databases and material properties

### 2. GPU-First Design

All performance-critical operations are designed for GPU execution:

- Vectorized tensor operations
- Batch processing of particles
- Coalesced memory access
- Minimal CPU-GPU data transfer

### 3. Modularity and Extensibility

Components are loosely coupled through well-defined interfaces:

- Easy to add new radionuclides
- Simple to extend physics models
- Straightforward to implement new output formats

### 4. Production-Ready Quality

Code follows best practices for reliability:

- Comprehensive error handling
- Input validation at all boundaries
- Detailed logging for debugging
- Clear error messages for users

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DosimetrySimulator                        │
│                  (Orchestration Layer)                       │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────────────┐
             │                                                 │
    ┌────────▼────────┐                              ┌────────▼────────┐
    │  InputManager   │                              │ GeometryProcessor│
    │  (Data I/O)     │                              │ (CT Processing)  │
    └────────┬────────┘                              └────────┬────────┘
             │                                                 │
             │                                                 │
    ┌────────▼────────────────────────────────────────────────▼────────┐
    │                    SourceTermProcessor                            │
    │              (Activity → Time-Integrated Activity)                │
    └────────┬──────────────────────────────────────────────────────────┘
             │
             │
    ┌────────▼────────────────────────────────────────────────────────┐
    │                    MonteCarloEngine                              │
    │                  (Particle Transport)                            │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
    │  │PhotonPhysics │  │ElectronPhysics│ │BetaSpectrum  │          │
    │  └──────────────┘  └──────────────┘  └──────────────┘          │
    └────────┬──────────────────────────────────────────────────────────┘
             │
             │
    ┌────────▼────────┐
    │ DoseSynthesis   │
    │ (Output)        │
    └─────────────────┘
```

### Data Flow

```
Input Images (NIfTI)
    │
    ├─→ CT Image ──────────→ GeometryProcessor ──→ Material Map
    │                                           ──→ Density Map
    │
    └─→ Activity Map ──────→ SourceTermProcessor ──→ TIA Maps
                                                  ──→ Primaries/Nuclide
         ↓
    MonteCarloEngine
         │
         ├─→ Sample Emission
         ├─→ Transport Particle
         ├─→ Accumulate Dose
         └─→ Generate Secondaries
         ↓
    DoseSynthesis
         │
         ├─→ Combine Nuclides
         ├─→ Calculate Uncertainty
         └─→ Export Results
         ↓
    Output (NIfTI or Objects)
```

## Module Descriptions

### Core Module (`MCGPURPTDosimetry/core/`)

#### DosimetrySimulator

**Purpose**: Main entry point and orchestration

**Responsibilities**:
- Initialize all components
- Coordinate simulation workflow
- Handle errors and logging
- Collect performance metrics

**Key Methods**:
```python
def __init__(self, config: SimulationConfig)
def run(self, ct_image, activity_map, ...) -> Dict
```

**Design Pattern**: Facade pattern - provides simple interface to complex subsystem

#### InputManager

**Purpose**: Medical image I/O and validation

**Responsibilities**:
- Load NIfTI files or nibabel objects
- Validate image compatibility (dimensions, spacing)
- Convert to PyTorch tensors
- Handle segmentation masks

**Key Methods**:
```python
def load_ct_image(self, source, device) -> torch.Tensor
def load_activity_image(self, source, device) -> torch.Tensor
def load_segmentation_masks(self, masks, device) -> Dict[str, torch.Tensor]
def validate_image_compatibility(self) -> None
```

**Design Pattern**: Adapter pattern - unifies file and object inputs

#### GeometryProcessor

**Purpose**: Convert CT to material/density maps

**Responsibilities**:
- Map Hounsfield units to materials
- Handle multi-range HU mapping (contrast enhancement)
- Apply segmentation masks with priority
- Extract density from CT or use material defaults

**Key Methods**:
```python
def create_geometry_data(self, ct_tensor, voxel_size, affine, ...) -> GeometryData
def _map_hu_to_materials(self, ct_tensor) -> torch.Tensor
def _apply_segmentation_masks(self, material_map, masks, priority) -> torch.Tensor
```

**Design Pattern**: Strategy pattern - configurable HU mapping strategies

#### SourceTermProcessor

**Purpose**: Calculate time-integrated activity and primary distribution

**Responsibilities**:
- Compute TIA from activity maps and half-lives
- Handle decay chains (parent → daughters)
- Distribute primaries proportional to TIA

**Key Methods**:
```python
def calculate_time_integrated_activity(self, activity_map, nuclide) -> Dict[str, torch.Tensor]
def calculate_num_primaries_per_nuclide(self, tia_maps, total_primaries) -> Dict[str, int]
```

**Design Pattern**: Service pattern - stateless computation

#### DoseSynthesis

**Purpose**: Combine dose contributions and calculate uncertainty

**Responsibilities**:
- Accumulate dose from multiple nuclides
- Track batch-wise dose for uncertainty
- Calculate per-voxel statistics
- Export results in requested format

**Key Methods**:
```python
def accumulate_nuclide_dose(self, nuclide, dose_map) -> None
def accumulate_batch_dose(self, dose_map) -> None
def calculate_uncertainty(self) -> torch.Tensor
def export_results(self, output_format, output_path) -> Dict
```

**Design Pattern**: Accumulator pattern - incremental result building

#### DataModels

**Purpose**: Define data structures for simulation

**Key Classes**:
- `GeometryData`: 3D geometry representation
- `ParticleStack`: Dynamic particle storage on GPU
- `SecondaryParticleBuffer`: Thread-safe secondary generation
- `EmissionData`, `DecayMode`, `NuclideData`: Decay data structures
- `CrossSectionData`, `StoppingPowerData`: Physics data structures

**Design Pattern**: Data Transfer Object (DTO) pattern

### Physics Module (`MCGPURPTDosimetry/physics/`)

#### MonteCarloEngine

**Purpose**: Main particle transport loop

**Responsibilities**:
- Initialize particle stacks
- Sample emission from activity distribution
- Dispatch particles to physics modules
- Manage secondary particle generation
- Accumulate dose to voxels

**Key Methods**:
```python
def simulate_nuclide(self, tia_map, nuclide, num_primaries) -> torch.Tensor
def _sample_emissions(self, tia_map, nuclide, num_primaries) -> ParticleStack
def _transport_photons(self, photon_stack) -> None
def _transport_electrons(self, electron_stack) -> None
def _handle_alphas(self, positions, energies, weights) -> None
```

**Design Pattern**: Template method pattern - defines transport skeleton

#### PhotonPhysics

**Purpose**: Photon interaction physics

**Responsibilities**:
- Sample interaction type (photoelectric, Compton, pair production)
- Calculate interaction distances
- Execute interactions and generate secondaries
- Handle characteristic X-rays

**Key Methods**:
```python
def sample_interaction_distance(self, energies, materials) -> torch.Tensor
def sample_interaction_type(self, energies, materials) -> torch.Tensor
def execute_photoelectric(self, photons) -> Tuple[electrons, xrays]
def execute_compton(self, photons) -> Tuple[photons_scattered, electrons]
def execute_pair_production(self, photons) -> Tuple[electrons, positrons]
```

**Design Pattern**: Strategy pattern - different interaction strategies

#### ElectronPhysics

**Purpose**: Electron/positron transport physics

**Responsibilities**:
- Calculate step size (energy and geometry limited)
- Apply continuous energy loss (CSDA)
- Simulate multiple Coulomb scattering
- Generate bremsstrahlung photons
- Handle positron annihilation

**Key Methods**:
```python
def calculate_step_size(self, electrons, geometry) -> torch.Tensor
def apply_energy_loss(self, electrons, step_size) -> torch.Tensor
def apply_multiple_scattering(self, electrons, step_size) -> torch.Tensor
def generate_bremsstrahlung(self, electrons, energy_loss) -> ParticleStack
def annihilate_positrons(self, positrons) -> ParticleStack
```

**Design Pattern**: Condensed history algorithm implementation

#### BetaSpectrum

**Purpose**: Beta decay energy spectrum sampling

**Responsibilities**:
- Precompute Fermi theory spectra
- Build alias tables for O(1) sampling
- Cache spectra for all nuclides

**Key Methods**:
```python
def compute_spectrum(self, endpoint_energy, atomic_number) -> np.ndarray
def build_alias_table(self, probabilities) -> Tuple[np.ndarray, np.ndarray]
def sample_energies(self, nuclide, num_samples) -> torch.Tensor
```

**Design Pattern**: Alias method for efficient sampling

#### DecayDatabase

**Purpose**: Load and query radionuclide decay data

**Responsibilities**:
- Parse JSON decay database
- Provide decay mode information
- List emissions for each nuclide
- Identify daughter nuclides

**Key Methods**:
```python
def load_database(self, path) -> None
def get_nuclide(self, name) -> NuclideData
def get_emissions(self, nuclide) -> List[EmissionData]
def get_daughters(self, nuclide) -> List[str]
```

**Design Pattern**: Repository pattern - data access abstraction

#### CrossSectionDatabase

**Purpose**: Load and interpolate interaction cross-sections

**Responsibilities**:
- Load HDF5 cross-section database
- Interpolate cross-sections at arbitrary energies
- Provide material-specific data
- Cache data on GPU for fast access

**Key Methods**:
```python
def load_database(self, path, device) -> None
def get_cross_sections(self, material, energies) -> Dict[str, torch.Tensor]
def get_stopping_powers(self, material, energies) -> Dict[str, torch.Tensor]
```

**Design Pattern**: Repository pattern with caching

### Utils Module (`MCGPURPTDosimetry/utils/`)

#### SimulationConfig

**Purpose**: Configuration management

**Responsibilities**:
- Define simulation parameters
- Validate configuration
- Load/save YAML configuration
- Provide default values

**Design Pattern**: Configuration object pattern

#### Logging

**Purpose**: Structured logging

**Responsibilities**:
- Set up file and console logging
- Provide logger instances
- Format log messages consistently

**Design Pattern**: Singleton logger pattern

#### Validation

**Purpose**: Input validation

**Responsibilities**:
- Validate configuration parameters
- Check image compatibility
- Verify physics database integrity

**Design Pattern**: Validator pattern

## Design Patterns Summary

### Creational Patterns

- **Factory Method**: Creating particle stacks and buffers
- **Singleton**: Logger instance

### Structural Patterns

- **Facade**: DosimetrySimulator provides simple interface
- **Adapter**: InputManager unifies file/object inputs
- **Repository**: Database classes abstract data access

### Behavioral Patterns

- **Strategy**: Configurable physics models and HU mapping
- **Template Method**: Monte Carlo transport loop
- **Observer**: Logging and progress reporting (future)

## GPU Programming Patterns

### Vectorized Operations

All particle operations are vectorized:

```python
# Bad: Loop over particles (CPU-style)
for i in range(len(particles)):
    distance[i] = -log(random()) / cross_section[i]

# Good: Vectorized operation (GPU-style)
distance = -torch.log(torch.rand_like(cross_section)) / cross_section
```

### Batch Processing

Process particles in large batches:

```python
# Process 100,000 particles simultaneously
batch_size = 100000
photon_stack = ParticleStack.create_empty(batch_size, device='cuda')
```

### Memory Coalescing

Access memory in contiguous patterns:

```python
# Good: Contiguous access
positions = particles.positions  # [N, 3] tensor
x = positions[:, 0]  # All x-coordinates

# Bad: Strided access (avoid)
x = positions[::2, 0]  # Every other particle
```

### Atomic Operations

Thread-safe dose accumulation:

```python
# Atomic add to dose map
torch.index_add_(dose_map, 0, voxel_indices, energy_deposits)
```

## Error Handling Strategy

### Validation at Boundaries

All public methods validate inputs:

```python
def run(self, ct_image, activity_map):
    # Validate inputs
    if ct_image is None:
        raise ValueError("ct_image cannot be None")
    
    # Validate compatibility
    self.input_manager.validate_image_compatibility()
```

### Graceful Degradation

Fall back to CPU if GPU unavailable:

```python
if self.device == 'cuda' and not torch.cuda.is_available():
    logger.warning("CUDA not available, falling back to CPU")
    self.device = 'cpu'
```

### Informative Error Messages

Provide actionable error messages:

```python
raise RuntimeError(
    f"GPU memory exhausted. Current: {max_particles} particles. "
    f"Try reducing max_particles_in_flight or use device='cpu'"
)
```

### Resource Cleanup

Ensure resources are released:

```python
try:
    results = mc_engine.simulate_nuclide(...)
finally:
    torch.cuda.empty_cache()  # Free GPU memory
```

## Performance Optimization Strategies

### 1. Minimize CPU-GPU Transfers

Keep data on GPU throughout simulation:

```python
# Good: All operations on GPU
geometry = GeometryData(
    material_map=material_map.to(device),
    density_map=density_map.to(device)
)

# Bad: Frequent transfers
for i in range(n):
    data_cpu = data_gpu.cpu()  # Slow!
    result = process(data_cpu)
    result_gpu = result.to(device)  # Slow!
```

### 2. Preallocate Memory

Avoid dynamic allocations in hot loops:

```python
# Good: Preallocate
particle_stack = ParticleStack.create_empty(capacity, device)

# Bad: Allocate in loop
for batch in batches:
    stack = ParticleStack.create_empty(len(batch), device)  # Slow!
```

### 3. Use In-Place Operations

Reduce memory allocations:

```python
# Good: In-place
energies -= energy_loss  # In-place subtraction

# Bad: Creates new tensor
energies = energies - energy_loss
```

### 4. Batch Similar Operations

Group operations for better GPU utilization:

```python
# Good: Batch all photons
transport_photons(all_photons)

# Bad: Process individually
for photon in photons:
    transport_photon(photon)
```

## Testing Strategy

### Unit Tests

Test individual components in isolation:

```python
def test_geometry_processor():
    processor = GeometryProcessor()
    ct_tensor = torch.randn(64, 64, 64)
    geometry = processor.create_geometry_data(ct_tensor, ...)
    assert geometry.material_map.shape == (64, 64, 64)
```

### Integration Tests

Test component interactions:

```python
def test_full_simulation():
    config = SimulationConfig(...)
    simulator = DosimetrySimulator(config)
    results = simulator.run(ct_image, activity_map)
    assert 'total_dose' in results
```

### Physics Validation

Compare against reference codes:

```python
def test_compton_scattering():
    # Compare against Geant4 results
    our_result = photon_physics.execute_compton(...)
    geant4_result = load_reference_data()
    assert np.allclose(our_result, geant4_result, rtol=0.05)
```

## Extension Points

### Adding New Radionuclides

1. Add decay data to JSON database
2. Regenerate physics databases
3. No code changes required

### Adding New Physics Models

1. Create new physics module (e.g., `neutron_physics.py`)
2. Implement required interface methods
3. Register in `MonteCarloEngine`

### Adding New Output Formats

1. Extend `DoseSynthesis.export_results()`
2. Add format-specific export logic
3. Update configuration validation

### Adding New Material Models

1. Add material data to HDF5 database
2. Update `GeometryProcessor` HU mapping
3. Regenerate cross-section database

## Code Organization Best Practices

### File Structure

```
MCGPURPTDosimetry/
├── __init__.py              # Package exports
├── core/                    # Core simulation logic
│   ├── __init__.py
│   ├── dosimetry_simulator.py
│   ├── input_manager.py
│   ├── geometry_processor.py
│   ├── source_term_processor.py
│   ├── dose_synthesis.py
│   └── data_models.py
├── physics/                 # Physics engines
│   ├── __init__.py
│   ├── monte_carlo_engine.py
│   ├── photon_physics.py
│   ├── electron_physics.py
│   ├── beta_spectrum.py
│   ├── decay_database.py
│   ├── cross_section_database.py
│   └── constants.py
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── validation.py
│   └── path_utils.py
└── physics_data/            # Data files
    ├── decay_databases/
    └── cross_section_databases/
```

### Naming Conventions

- **Classes**: PascalCase (`DosimetrySimulator`)
- **Functions**: snake_case (`calculate_dose`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_PARTICLES`)
- **Private methods**: Leading underscore (`_internal_method`)

### Documentation Standards

- Docstrings for all public classes and methods
- Type hints for function signatures
- Inline comments for complex algorithms
- README files in each major directory

## Future Architecture Enhancements

### Planned Improvements

1. **Plugin System**: Dynamic loading of physics modules
2. **Parallel Batch Processing**: Multiple simulations simultaneously
3. **Distributed Computing**: Multi-GPU and multi-node support
4. **Streaming I/O**: Process large datasets without loading entirely
5. **Checkpoint/Resume**: Save and restore simulation state

### Research Directions

1. **Machine Learning Integration**: Surrogate models for ultra-fast dose estimation
2. **Adaptive Sampling**: Importance sampling for variance reduction
3. **Hybrid CPU-GPU**: Optimal work distribution
4. **Just-In-Time Compilation**: Further performance optimization

---

**Next**: See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for contributing to the codebase.
