# Mask-Based Tissue Definition Workflow

## Overview

The mask-based tissue definition feature allows you to directly specify tissue regions using segmentation masks, providing an alternative or complement to HU-based material assignment. This is particularly useful for:

- **Organ-specific dosimetry**: Accurate dose calculation to specific organs
- **Tumor delineation**: Explicit definition of tumor regions
- **Bone segmentation**: Separate cortical and trabecular bone
- **Treatment planning integration**: Use contours from radiotherapy planning systems
- **Research applications**: Test sensitivity to material composition assumptions

## Key Features

### 1. Flexible Input Formats

Masks can be provided in multiple formats:

- **Binary masks** (0/1) for single tissue type
- **Multi-label masks** with integer labels for multiple tissues
- **File paths** (NIfTI format): `'path/to/mask.nii.gz'`
- **nibabel objects**: `nib.Nifti1Image` objects
- **Dictionary-based specification**: `{'Liver': liver_mask, 'Tumor': tumor_mask}`

### 2. Hybrid Material Assignment

Combine HU-based assignment with mask-defined regions:

- Masks **override** HU-based assignment in specified regions
- Unmasked regions use standard HU-to-material lookup
- Configurable priority order for overlapping masks

### 3. Density Handling Options

Two modes for handling density in masked regions:

- **CT-derived density** (default): Preserves anatomical density variations within organs
- **Fixed material density**: Uses database density for uniform organ properties

### 4. Integration with Segmentation Workflows

- Compatible with deep learning segmentation outputs
- Support for DICOM RT Structure Sets (via conversion to masks)
- Integration with treatment planning system contours
- Multi-organ segmentation from medical imaging AI

## Usage Examples

### Basic Usage

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
import nibabel as nib

# Load CT and activity images
ct_img = nib.load('patient_ct.nii.gz')
activity_img = nib.load('patient_activity.nii.gz')

# Load segmentation masks
liver_mask = nib.load('segmentation/liver.nii.gz')
tumor_mask = nib.load('segmentation/tumor.nii.gz')

# Create mask dictionary
tissue_masks = {
    'Liver': liver_mask,
    'Tumor': tumor_mask
}

# Configure simulation
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000
)

# Run simulation with masks
simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=tissue_masks  # Masks override HU-based assignment
)
```

### Priority Order for Overlapping Masks

When masks overlap, specify priority order (last has highest priority):

```python
# Define masks
tissue_masks = {
    'Liver': liver_mask,
    'Kidney_Left': kidney_left_mask,
    'Kidney_Right': kidney_right_mask,
    'Tumor': tumor_mask  # Overlaps with liver
}

# Priority order: organs first, then tumor
priority_order = ['Liver', 'Kidney_Left', 'Kidney_Right', 'Tumor']

# Run simulation
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=tissue_masks,
    mask_priority_order=priority_order  # Tumor overrides liver in overlaps
)
```

### Hybrid HU + Mask-Based Assignment

Use masks for specific organs, HU-based assignment for the rest:

```python
# Only define tumor and bone explicitly
selected_masks = {
    'Tumor': tumor_mask,
    'Bone_Cortical': cortical_bone_mask
}

# Run simulation
# Masked regions: explicit material assignment
# Unmasked regions: HU-based assignment
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=selected_masks
)
```

### Density Handling Options

```python
# Option 1: Use CT-derived density (default)
# Preserves anatomical density variations within organs
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=tissue_masks,
    use_ct_density=True  # Default
)

# Option 2: Use fixed material density from database
# Uniform density within each organ
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=tissue_masks,
    use_ct_density=False
)
```

### File-Based Masks

```python
# Specify masks as file paths
mask_paths = {
    'Liver': 'segmentation/liver_mask.nii.gz',
    'Tumor': 'segmentation/tumor_mask.nii.gz',
    'Kidney_Left': 'segmentation/kidney_left_mask.nii.gz',
    'Kidney_Right': 'segmentation/kidney_right_mask.nii.gz'
}

# Run simulation with file paths
results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz',
    tissue_masks=mask_paths
)
```

## Use Case Examples

### 1. Organ-Specific Dosimetry

Calculate dose to specific organs using segmentation:

```python
# Load organ segmentation from medical imaging AI
organ_masks = {
    'Liver': nib.load('segmentation/liver.nii.gz'),
    'Kidney_Left': nib.load('segmentation/kidney_left.nii.gz'),
    'Kidney_Right': nib.load('segmentation/kidney_right.nii.gz'),
    'Spleen': nib.load('segmentation/spleen.nii.gz')
}

results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=organ_masks
)

# Calculate organ-specific doses
dose_array = results['total_dose'].get_fdata()
for organ_name, mask_img in organ_masks.items():
    mask_data = mask_img.get_fdata()
    organ_dose = dose_array * mask_data
    mean_dose = np.mean(organ_dose[mask_data > 0])
    print(f"{organ_name} mean dose: {mean_dose:.2f} Gy")
```

### 2. Tumor Dosimetry

Define tumor regions with specific material properties:

```python
# Define tumor and metastases
tumor_masks = {
    'Tumor_Primary': primary_tumor_mask,
    'Tumor_Metastasis_1': met1_mask,
    'Tumor_Metastasis_2': met2_mask
}

# Tumors override surrounding tissue assignment
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=tumor_masks
)
```

### 3. Bone Segmentation

Separate cortical and trabecular bone for accurate dosimetry:

```python
# Segment bone types
bone_masks = {
    'Bone_Cortical': cortical_bone_mask,
    'Bone_Trabecular': trabecular_bone_mask
}

# More accurate than HU-based bone classification
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=bone_masks
)
```

### 4. Treatment Planning Integration

Import contours from radiotherapy planning systems:

```python
# Convert DICOM RT Structures to NIfTI masks (external tool)
# Then load for dosimetry simulation

planning_masks = {
    'GTV': gross_tumor_volume_mask,      # Gross Tumor Volume
    'CTV': clinical_target_volume_mask,  # Clinical Target Volume
    'PTV': planning_target_volume_mask,  # Planning Target Volume
    'Liver': liver_contour_mask,
    'Kidney_Left': left_kidney_contour_mask,
    'Kidney_Right': right_kidney_contour_mask
}

# Priority order: organs first, then tumor volumes
priority_order = ['Liver', 'Kidney_Left', 'Kidney_Right', 'GTV', 'CTV', 'PTV']

results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=planning_masks,
    mask_priority_order=priority_order
)
```

## Mask Requirements

### Spatial Compatibility

- Mask dimensions **must match** CT image dimensions
- Mask affine matrix should match CT affine matrix
- Voxel spacing should be identical

### Mask Values

- **Binary masks**: 0 (background) and 1 (tissue region)
- **Multi-label masks**: Integer labels (0 = background, 1+ = tissue types)
- Non-binary values: All non-zero values treated as mask region (with warning)

### Tissue Names

Tissue names in mask dictionary must correspond to known materials:

**Available materials:**
- `Air`
- `Lung`
- `Muscle`
- `Soft_Tissue`
- `Fat`
- `Bone_Cortical`
- `Bone_Trabecular`
- `Bone_Generic`
- `Bone`
- `Water`
- `Iodine_Contrast_Mixture`
- `Liver` (uses Soft_Tissue properties)
- `Kidney_Left`, `Kidney_Right` (use Soft_Tissue properties)
- `Tumor` (uses Soft_Tissue properties)

## Validation and Error Handling

The system automatically validates masks:

```python
# Validation checks:
# 1. Spatial dimensions match CT image
# 2. Mask values are valid (0/1 for binary)
# 3. Tissue names correspond to known materials
# 4. No invalid overlaps (if specified)

try:
    results = simulator.run(
        ct_image=ct_img,
        activity_map=activity_img,
        tissue_masks=tissue_masks
    )
except ImageDimensionMismatchError as e:
    print(f"Mask dimension mismatch: {e}")
except InvalidMaterialError as e:
    print(f"Unknown tissue type: {e}")
```

## Advantages of Mask-Based Definition

1. **Accuracy**: Direct specification eliminates HU-based misclassification
2. **Flexibility**: Support for custom tissue types and compositions
3. **Integration**: Seamless workflow with segmentation AI and planning systems
4. **Reproducibility**: Consistent material assignment across studies
5. **Clinical relevance**: Align with clinical organ delineations
6. **Research utility**: Test sensitivity to material composition assumptions

## Performance Considerations

- Mask processing performed on GPU for efficiency
- Minimal overhead compared to HU-based assignment
- Memory footprint: ~4 bytes per voxel per mask
- **Recommended**: Limit to 10-20 masks for typical clinical cases
- Large numbers of masks may impact memory usage

## Best Practices

### 1. Mask Quality

- Use high-quality segmentation (manual or validated AI)
- Verify mask alignment with CT image
- Check for gaps or overlaps in segmentation

### 2. Priority Order

- Place background organs first in priority order
- Place specific structures (tumors, lesions) last
- Last mask in priority order wins in overlapping regions

### 3. Density Handling

- Use `use_ct_density=True` (default) for most cases
- Use `use_ct_density=False` for idealized organ models
- Consider anatomical density variations within organs

### 4. Validation

- Always validate mask compatibility before large simulations
- Check organ dose statistics for reasonableness
- Compare with HU-based results for consistency

### 5. Documentation

- Document mask sources (manual, AI, planning system)
- Record segmentation parameters and validation
- Include mask metadata in simulation records

## Troubleshooting

### Common Issues

**Issue**: `ImageDimensionMismatchError`
- **Cause**: Mask dimensions don't match CT image
- **Solution**: Resample mask to CT image space

**Issue**: `InvalidMaterialError`
- **Cause**: Unknown tissue name in mask dictionary
- **Solution**: Use valid material names from available materials list

**Issue**: Unexpected dose distribution
- **Cause**: Mask overlap or priority order issues
- **Solution**: Check mask overlaps and adjust priority order

**Issue**: High memory usage
- **Cause**: Too many masks loaded simultaneously
- **Solution**: Reduce number of masks or process in batches

## Additional Resources

- **Example script**: `examples/mask_based_tissue_definition.py`
- **API documentation**: See `DosimetrySimulator.run()` method
- **Design document**: `.kiro/specs/gpu-dosimetry-system/design.md`
- **Requirements**: `.kiro/specs/gpu-dosimetry-system/requirements.md`

## Support

For questions or issues with mask-based tissue definition:
1. Check this documentation and examples
2. Review validation error messages
3. Consult API documentation
4. Open an issue on the project repository
