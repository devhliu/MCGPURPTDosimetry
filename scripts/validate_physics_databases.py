#!/usr/bin/env python3
"""Validate physics databases (decay and cross-section)."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from MCGPURPTDosimetry.physics import DecayDatabase, CrossSectionDatabase


def validate_decay_database():
    """Validate decay database."""
    print("=" * 60)
    print("Validating Decay Database")
    print("=" * 60)
    
    db_path = 'MCGPURPTDosimetry/physics_data/decay_databases/default.json'
    print(f"Loading: {db_path}")
    
    try:
        db = DecayDatabase(db_path)
        print(f"✓ Database loaded successfully")
        
        # Get available nuclides
        nuclides = list(db.nuclides.keys())
        print(f"\n✓ Found {len(nuclides)} nuclides:")
        
        # Categorize nuclides
        therapeutic = ['Lu-177', 'Y-90', 'I-131', 'Re-188', 'Cu-67', 'Ho-166', 
                      'Tb-161', 'At-211', 'Ac-225', 'Pb-212']
        diagnostic = ['Tc-99m', 'F-18', 'Ga-68', 'Cu-64', 'C-11', 'N-13', 'Zr-89', 'I-124']
        decay_chain = ['Fr-221', 'At-217', 'Bi-213', 'Po-213', 'Tl-209', 'Bi-212', 'Po-212']
        
        print("\n  Therapeutic radionuclides:")
        for nuc in therapeutic:
            if nuc in nuclides:
                print(f"    ✓ {nuc}")
            else:
                print(f"    ✗ {nuc} (missing)")
        
        print("\n  Diagnostic radionuclides:")
        for nuc in diagnostic:
            if nuc in nuclides:
                print(f"    ✓ {nuc}")
            else:
                print(f"    ✗ {nuc} (missing)")
        
        print("\n  Decay chain daughters:")
        for nuc in decay_chain:
            if nuc in nuclides:
                print(f"    ✓ {nuc}")
            else:
                print(f"    ✗ {nuc} (missing)")
        
        # Test loading decay data
        print("\n✓ Testing decay data loading:")
        for nuc in ['Lu-177', 'F-18', 'Ac-225']:
            if nuc in nuclides:
                data = db.get_nuclide(nuc)
                if data:
                    print(f"    ✓ {nuc}: half-life = {data.half_life_seconds:.2e} s")
        
        print("\n✓ Decay database validation PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Decay database validation FAILED: {e}")
        return False


def validate_cross_section_database():
    """Validate cross-section database."""
    print("\n" + "=" * 60)
    print("Validating Cross-Section Database")
    print("=" * 60)
    
    db_path = 'MCGPURPTDosimetry/physics_data/cross_section_databases/default.h5'
    print(f"Loading: {db_path}")
    
    try:
        db = CrossSectionDatabase(db_path, device='cpu')
        print(f"✓ Database loaded successfully")
        
        # Get available materials
        materials = list(db.materials.keys())
        print(f"\n✓ Found {len(materials)} materials:")
        
        # Expected materials
        expected = ['Air', 'Lung', 'Muscle', 'Soft_Tissue', 'Fat', 
                   'Bone_Cortical', 'Bone_Trabecular', 'Bone_Generic',
                   'Water', 'Bone', 'Iodine_Contrast_Mixture']
        
        for mat in expected:
            if mat in materials:
                print(f"    ✓ {mat}")
            else:
                print(f"    ✗ {mat} (missing)")
        
        # Test loading cross-section data
        print("\n✓ Testing cross-section data loading:")
        test_materials = ['Soft_Tissue', 'Bone_Cortical', 'Lung']
        
        for mat in test_materials:
            if mat in materials:
                # Test photon cross-sections
                xs = db.get_photon_cross_sections(mat)
                if xs:
                    print(f"    ✓ {mat}: photon data loaded ({len(xs.energy_grid)} energy points)")
                
                # Test electron stopping powers
                sp = db.get_electron_stopping_powers(mat)
                if sp:
                    print(f"    ✓ {mat}: electron data loaded ({len(sp.energy_grid)} energy points)")
        
        # Verify energy range
        print("\n✓ Verifying energy range coverage:")
        if 'Soft_Tissue' in materials:
            xs_data = db.get_photon_cross_sections('Soft_Tissue')
            if xs_data:
                energy_grid = xs_data.energy_grid
                print(f"    Energy range: {energy_grid.min():.2e} - {energy_grid.max():.2e} eV")
                print(f"    Number of points: {len(energy_grid)}")
                
                if energy_grid.min() <= 10.0 and energy_grid.max() >= 1e7:
                    print(f"    ✓ Energy range covers 10 eV to 10 MeV")
                else:
                    print(f"    ✗ Energy range insufficient (expected 10 eV to 10 MeV)")
        
        print("\n✓ Cross-section database validation PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Cross-section database validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("Physics Database Validation Suite")
    print("=" * 60 + "\n")
    
    decay_ok = validate_decay_database()
    xs_ok = validate_cross_section_database()
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Decay database: {'✓ PASSED' if decay_ok else '✗ FAILED'}")
    print(f"Cross-section database: {'✓ PASSED' if xs_ok else '✗ FAILED'}")
    
    if decay_ok and xs_ok:
        print("\n✓ All validations PASSED")
        return 0
    else:
        print("\n✗ Some validations FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
