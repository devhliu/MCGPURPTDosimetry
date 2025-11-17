# API Reference

## Core Classes

### DosimetrySimulator

Main orchestration class for GPU-accelerated dosimetry simulation.

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

simulator = DosimetrySimulator(config)
```

#### Constructor

```python
DosimetrySimulator(config: SimulationConfig)
```

**Parameters:**
- `config` (SimulationConfig): Simulation configuration object

**Attributes:**
- `config`: Simulation configuration
- `input_manager`: Input image manager
- `geometry_processor`: Geometry processor
- `logger`: Logger instance

#### Methods

##### run()

Run complete dosimetry simulation.

```python
results = simulator.run(
    ct_image,
    activity_map,
    tissue_masks=None,
    mask_priority_order=None,
    use_ct_density=True
)
```

**Parameters:**
- `ct_image` (str | nib.Nifti1Image): CT image (file path or nibabel object)
- `activity_map` (str | nib.Nifti1Image): Activity map in Bq/pixel (file path or nibabel object)
- `tissue_masks` (Dict[str, str | nib.Nifti1Image], optional): Dictionary mapping tissue names to mask sources
- `mask_priority_order` (List[str], optional): Priority order for applying overlapping masks (last has highest priority)
- `use_ct_density` (bool, default=True): If True, use CT-derived density in masked regions; if False, use material database density

**Returns:**
- `Dict`: Dictionary with simulation results
  - `'total_dose'`: Total dose map (nibabel image or file path)
  - `'uncertainty'`: Statistical uncertainty map
  - `'{nuclide}_dose'`: Per-nuclide dose maps
  - `'statistics'`: Dose statistics dictionary
  - `'performance'`: Performance metrics

**Example:**
```python
results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz'
)

# With masks
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks={'Liver': liver_mask, 'Tumor': tumor_mask},
    mask_priority_order=['Liver', 'Tumor']
)
```

---

### SimulationConfig

Configuration class for dosimetry simulations.

```python
from MCGPURPTDosimetry import SimulationConfig

config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000
)
```

#### Constructor

```python
SimulationConfig(
    radionuclide: str,
    num_primaries: int,
    energy_cutoff_keV: float = 10.0,
    num_batches: int = 10,
    output_format: str = 'object',
    output_path: Optional[str] = None,
    device: str = 'cuda',
    random_seed: Optional[int] = None,
    decay_database_path: Optional[str] = None,
    cross_section_database_path: Optional[str] = None,
    hu_to_material_lut: Optional[Dict] = None,
    max_particles_in_flight: int = 100000
)
```

**Parameters:**
- `radionuclide` (str): Radionuclide name (e.g., 'Lu-177', 'I-131', 'Y-90')
- `num_primaries` (int): Number of primary particles to simulate
- `energy_cutoff_keV` (float, default=10.0): Energy cutoff in keV
- `num_batches` (int, default=10): Number of batches for uncertainty estimation
- `output_format` (str, default='object'): Output format ('object' or 'file')
- `output_path` (str, optional): Output directory path (required if output_format='file')
- `device` (str, default='cuda'): Device for computation ('cuda' or 'cpu')
- `random_seed` (int, optional): Random seed for reproducibility
- `decay_database_path` (str, optional): Path to decay database (uses default if None)
- `cross_section_database_path` (str, optional): Path to cross-section database (uses default if None)
- `hu_to_material_lut` (Dict, optional): HU-to-material lookup table (uses default if None)
- `max_particles_in_flight` (int, default=100000): Maximum particles in flight simultaneously

#### Class Methods

##### get_default_config()

Get default configuration.

```python
config = SimulationConfig.get_default_config()
```

**Returns:**
- `SimulationConfig`: Default configuration object

##### from_yaml()

Load configuration from YAML file.

```python
config = SimulationConfig.from_yaml('config.yaml')
```

**Parameters:**
- `yaml_path` (str): Path to YAML configuration file

**Returns:**
- `SimulationConfig`: Configuration object

#### Instance Methods

##### to_yaml()

Save configuration to YAML file.

```python
config.to_yaml('config.yaml')
```

**Parameters:**
- `yaml_path` (str): Path to output YAML file

---

## Physics Classes

### DecayDatabase

Manages nuclide decay data from JSON database.

```python
from MCGPURPTDosimetry.physics import DecayDatabase

decay_db = DecayDatabase('path/to/decay_database.json')
```

#### Constructor

```python
DecayDatabase(database_path: str)
```

**Parameters:**
- `database_path` (str): Path to JSON decay database file

#### Methods

##### get_nuclide()

Get nuclide data by name.

```python
nuclide_data = decay_db.get_nuclide('Lu-177')
```

**Parameters:**
- `nuclide_name` (str): Name of nuclide

**Returns:**
- `NuclideData | None`: Nuclide data object or None if not found

##### get_decay_chain()

Resolve complete decay chain for a parent nuclide.

```python
chain = decay_db.get_decay_chain('Ac-225', max_depth=10)
```

**Parameters:**
- `parent_nuclide` (str): Parent nuclide name
- `max_depth` (int, default=10): Maximum chain depth

**Returns:**
- `List[str]`: List of nuclide names in decay chain

##### get_all_emissions()

Get all emissions for a nuclide across all decay modes.

```python
emissions = decay_db.get_all_emissions('Lu-177')
```

**Parameters:**
- `nuclide_name` (str): Name of nuclide

**Returns:**
- `List[Emission]`: List of emission objects

---

### CrossSectionDatabase

Manages photon and electron cross-section data from HDF5 database.

```python
from MCGPURPTDosimetry.physics import CrossSectionDatabase

xs_db = CrossSectionDatabase('path/to/cross_sections.h5', device='cuda')
```

#### Constructor

```python
CrossSectionDatabase(database_path: str, device: str = 'cuda')
```

**Parameters:**
- `database_path` (str): Path to HDF5 cross-section database
- `device` (str, default='cuda'): Device for tensor storage

#### Methods

##### get_photon_cross_sections()

Get photon cross-section data for a material.

```python
xs_data = xs_db.get_photon_cross_sections('Soft_Tissue')
```

**Parameters:**
- `material_name` (str): Name of material

**Returns:**
- `CrossSectionData | None`: Cross-section data object or None if not found

**CrossSectionData attributes:**
- `energy_grid`: Energy grid (eV)
- `photoelectric`: Photoelectric cross-section (cm²/g)
- `compton`: Compton cross-section (cm²/g)
- `pair_production`: Pair production cross-section (cm²/g)
- `total`: Total cross-section (cm²/g)

##### get_electron_stopping_powers()

Get electron stopping power data for a material.

```python
sp_data = xs_db.get_electron_stopping_powers('Soft_Tissue')
```

**Parameters:**
- `material_name` (str): Name of material

**Returns:**
- `StoppingPowerData | None`: Stopping power data object or None if not found

**StoppingPowerData attributes:**
- `energy_grid`: Energy grid (eV)
- `collisional`: Collisional stopping power (MeV·cm²/g)
- `radiative`: Radiative stopping power (MeV·cm²/g)
- `total`: Total stopping power (MeV·cm²/g)
- `density_effect`: Density effect correction

---

## Data Preparation Classes

### DecayDatabaseGenerator

Generates nuclide decay databases from ICRP-107 data.

```python
from MCGPURPTDosimetry.physics_data_preparation import DecayDatabaseGenerator

generator = DecayDatabaseGenerator('path/to/icrp107_data/')
```

#### Constructor

```python
DecayDatabaseGenerator(icrp107_data_path: str)
```

**Parameters:**
- `icrp107_data_path` (str): Path to ICRP-107 data directory

#### Methods

##### parse_icrp107()

Parse ICRP-107 data for specified nuclides.

```python
nuclide_data = generator.parse_icrp107(['Lu-177', 'I-131', 'Y-90'])
```

**Parameters:**
- `nuclide_list` (List[str]): List of nuclide names to parse

**Returns:**
- `Dict[str, dict]`: Dictionary of parsed nuclide data

##### generate_database()

Generate JSON decay database file.

```python
generator.generate_database('output/decay_database.json')
```

**Parameters:**
- `output_path` (str): Path to output JSON file

##### validate_database()

Validate generated database.

```python
is_valid = generator.validate_database('output/decay_database.json')
```

**Parameters:**
- `database_path` (str): Path to database file

**Returns:**
- `bool`: True if valid

---

### CrossSectionGenerator

Generates cross-section databases for materials.

```python
from MCGPURPTDosimetry.physics_data_preparation import CrossSectionGenerator

xs_gen = CrossSectionGenerator(physics_backend='geant4')
```

#### Constructor

```python
CrossSectionGenerator(physics_backend: str = 'geant4')
```

**Parameters:**
- `physics_backend` (str, default='geant4'): Physics calculation backend ('geant4' or 'penelope')

#### Methods

##### define_material()

Define a material for cross-section calculation.

```python
xs_gen.define_material(
    'Soft_Tissue',
    composition={'H': 0.105, 'C': 0.256, 'N': 0.027, 'O': 0.602},
    density=1.04
)
```

**Parameters:**
- `name` (str): Material name
- `composition` (Dict[str, float]): Elemental composition {element: mass_fraction}
- `density` (float): Density in g/cm³

##### calculate_cross_sections()

Calculate cross-sections for materials.

```python
energy_grid = np.logspace(1, 7, 1000)  # 10 eV to 10 MeV
xs_gen.calculate_cross_sections(energy_grid, ['Soft_Tissue', 'Bone'])
```

**Parameters:**
- `energy_grid` (np.ndarray): Energy points in eV
- `materials` (List[str]): List of material names

##### export_database()

Export cross-section database to HDF5.

```python
xs_gen.export_database('output/cross_sections.h5')
```

**Parameters:**
- `output_path` (str): Path to output HDF5 file

---

## Utility Classes

### InputManager

Manages flexible input of medical images (files or objects).

```python
from MCGPURPTDosimetry.core import InputManager

input_mgr = InputManager()
```

#### Methods

##### load_ct_image()

Load CT anatomical image from file or object.

```python
ct_tensor = input_mgr.load_ct_image('patient_ct.nii.gz', device='cuda')
# or
ct_tensor = input_mgr.load_ct_image(ct_nifti_object, device='cuda')
```

**Parameters:**
- `source` (str | nib.Nifti1Image): File path or nibabel image object
- `device` (str, default='cuda'): Device to load tensor to

**Returns:**
- `torch.Tensor`: CT data as PyTorch tensor [X, Y, Z]

##### load_activity_image()

Load activity map from file or object.

```python
activity_tensor = input_mgr.load_activity_image('activity.nii.gz', device='cuda')
```

**Parameters:**
- `source` (str | nib.Nifti1Image): File path or nibabel image object
- `device` (str, default='cuda'): Device to load tensor to

**Returns:**
- `torch.Tensor`: Activity data as PyTorch tensor [X, Y, Z]

##### load_segmentation_masks()

Load segmentation masks from files or objects.

```python
mask_tensors = input_mgr.load_segmentation_masks(
    {'Liver': 'liver_mask.nii.gz', 'Tumor': tumor_mask_object},
    device='cuda'
)
```

**Parameters:**
- `mask_sources` (Dict[str, str | nib.Nifti1Image]): Dictionary mapping tissue names to mask sources
- `device` (str, default='cuda'): Device to load tensors to

**Returns:**
- `Dict[str, torch.Tensor]`: Dictionary of mask tensors

##### validate_image_compatibility()

Validate that loaded images are spatially compatible.

```python
input_mgr.validate_image_compatibility()
```

**Raises:**
- `ImageDimensionMismatchError`: If images have incompatible dimensions

---

### GeometryProcessor

Processes CT images to create material and density maps.

```python
from MCGPURPTDosimetry.core import GeometryProcessor

geom_processor = GeometryProcessor(hu_to_material_lut=config.hu_to_material_lut)
```

#### Constructor

```python
GeometryProcessor(
    hu_to_material_lut: Dict[str, List[Tuple[float, float]]],
    material_properties: Optional[Dict[int, MaterialProperties]] = None
)
```

**Parameters:**
- `hu_to_material_lut`: Dictionary mapping material names to list of HU ranges
- `material_properties` (optional): Dictionary of material properties by ID

#### Methods

##### create_geometry_data()

Create complete geometry data from CT image.

```python
geometry = geom_processor.create_geometry_data(
    ct_tensor,
    voxel_size=(2.0, 2.0, 2.0),
    affine_matrix=affine,
    mask_dict=tissue_masks,
    mask_priority_order=['Liver', 'Tumor'],
    use_ct_density=True
)
```

**Parameters:**
- `ct_tensor` (torch.Tensor): CT data in Hounsfield Units
- `voxel_size` (Tuple[float, float, float]): Voxel dimensions in mm (dx, dy, dz)
- `affine_matrix` (np.ndarray): 4x4 spatial transform matrix
- `mask_dict` (Dict[str, torch.Tensor], optional): Dictionary of tissue masks
- `mask_priority_order` (List[str], optional): Priority order for overlapping masks
- `use_ct_density` (bool, default=True): If True, use CT-derived density in masked regions

**Returns:**
- `GeometryData`: Complete geometry data object

---

## Data Models

### GeometryData

Container for geometry data.

**Attributes:**
- `material_map` (torch.Tensor): Material ID map [X, Y, Z]
- `density_map` (torch.Tensor): Density map in g/cm³ [X, Y, Z]
- `voxel_size` (Tuple[float, float, float]): Voxel dimensions in mm
- `dimensions` (Tuple[int, int, int]): Grid dimensions (nx, ny, nz)
- `affine_matrix` (np.ndarray): 4x4 spatial transform matrix

### NuclideData

Container for nuclide decay data.

**Attributes:**
- `name` (str): Nuclide name
- `half_life_seconds` (float): Half-life in seconds
- `decay_modes` (Dict): Dictionary of decay modes

### MaterialProperties

Container for material properties.

**Attributes:**
- `material_id` (int): Material ID
- `name` (str): Material name
- `density` (float): Density in g/cm³
- `composition` (Dict[str, float]): Elemental composition

---

## Available Radionuclides

### Therapeutic Radionuclides

- **Lu-177**: Beta-emitter for PRRT (6.75 days)
- **Y-90**: High-energy beta-emitter (2.67 days)
- **I-131**: Beta/gamma-emitter (8.02 days)
- **Re-188**: Beta-emitter for bone pain palliation (17 hours)
- **Cu-67**: Beta-emitter for radioimmunotherapy (2.58 days)
- **Ho-166**: Beta-emitter for radioembolization (26.8 hours)
- **Tb-161**: Beta-emitter for targeted therapy (6.89 days)
- **At-211**: Alpha-emitter for targeted alpha therapy (7.2 hours)
- **Ac-225**: Alpha-emitter with decay chain (10 days)
- **Pb-212**: Alpha-emitter via daughter Bi-212 (10.6 hours)

### Diagnostic Radionuclides

- **Tc-99m**: Gamma-emitter for SPECT (6.01 hours)
- **F-18**: Positron-emitter for PET (109.77 minutes)
- **Ga-68**: Positron-emitter for PET (67.71 minutes)
- **Cu-64**: Positron-emitter for PET (12.7 hours)
- **C-11**: Positron-emitter for PET (20.38 minutes)
- **N-13**: Positron-emitter for PET (9.97 minutes)
- **Zr-89**: Positron-emitter for immuno-PET (78.41 hours)
- **I-124**: Positron-emitter for PET (4.18 days)

---

## Available Materials

### Human Body Tissues

- **Air**: Respiratory airways (ρ = 0.0012 g/cm³)
- **Lung**: Inflated lung tissue (ρ = 0.26 g/cm³)
- **Muscle**: Skeletal and cardiac muscle (ρ = 1.05 g/cm³)
- **Soft_Tissue**: Generic soft tissue (ρ = 1.04 g/cm³)
- **Fat**: Adipose tissue (ρ = 0.95 g/cm³)
- **Bone_Cortical**: Compact bone (ρ = 1.92 g/cm³)
- **Bone_Trabecular**: Spongy bone (ρ = 1.18 g/cm³)
- **Bone_Generic**: Average bone properties (ρ = 1.55 g/cm³)
- **Bone**: Legacy bone material (ρ = 1.85 g/cm³)
- **Water**: Water (ρ = 1.0 g/cm³)
- **Iodine_Contrast_Mixture**: Contrast-enhanced tissue (ρ = 1.15 g/cm³)

---

## Error Handling

### Common Exceptions

#### InvalidImageFormatError

Raised when input image format is invalid.

```python
try:
    ct_tensor = input_mgr.load_ct_image(invalid_source)
except InvalidImageFormatError as e:
    print(f"Invalid image format: {e}")
```

#### ImageDimensionMismatchError

Raised when image dimensions don't match.

```python
try:
    input_mgr.validate_image_compatibility()
except ImageDimensionMismatchError as e:
    print(f"Dimension mismatch: {e}")
```

#### InvalidMaterialError

Raised when material name is not recognized.

```python
try:
    results = simulator.run(ct_image, activity_map, tissue_masks={'Unknown': mask})
except InvalidMaterialError as e:
    print(f"Unknown material: {e}")
```

---

## Performance Tips

### GPU Memory Management

```python
# Use smaller batch sizes for limited GPU memory
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    max_particles_in_flight=50000  # Reduce if out of memory
)
```

### CPU Fallback

```python
# Use CPU if CUDA not available
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100_000,  # Reduce for CPU
    device='cpu'
)
```

### Batch Processing

```python
# Process large datasets in batches
for batch_idx in range(num_batches):
    results = simulator.run(
        ct_image=batch_ct[batch_idx],
        activity_map=batch_activity[batch_idx]
    )
    # Process results...
```

---

## See Also

- [User Guide](USER_GUIDE.md)
- [Mask-Based Workflow](MASK_BASED_WORKFLOW.md)
- [Examples](../examples/)
- [Design Document](DESIGN_ARCHITECTURE.md)
