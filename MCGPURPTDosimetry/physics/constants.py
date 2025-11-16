"""Physical constants for Monte Carlo simulation."""

# Fundamental constants
ELECTRON_REST_MASS_KEV = 511.0  # Electron rest mass energy in keV
SPEED_OF_LIGHT = 299792458.0  # Speed of light in m/s

# Numerical constants
EPSILON = 1e-30  # Small number to avoid division by zero
TOLERANCE = 1e-6  # Numerical tolerance for comparisons

# Energy thresholds
DEFAULT_ENERGY_CUTOFF_KEV = 10.0  # Default energy cutoff in keV
ANNIHILATION_PHOTON_ENERGY_KEV = 511.0  # Positron annihilation photon energy

# Conversion factors
KEV_TO_JOULES = 1.602e-16  # Conversion from keV to Joules
MM_TO_CM = 0.1  # Conversion from mm to cm
CM_TO_MM = 10.0  # Conversion from cm to mm

# Physics parameters
MAX_ENERGY_LOSS_FRACTION = 0.1  # Maximum fractional energy loss per step
STRAGGLING_FACTOR = 0.1  # Energy straggling factor (simplified)
HIGHLAND_CONSTANT = 13.6  # Highland formula constant in MeV

# Material properties
SOFT_TISSUE_Z = 7.4  # Effective atomic number for soft tissue
SOFT_TISSUE_A = 7.4  # Average mass number for soft tissue
WATER_DENSITY = 1.0  # Water density in g/cm³

# Sampling parameters
MAX_REJECTION_ITERATIONS = 100  # Maximum iterations for rejection sampling
BREMSSTRAHLUNG_THRESHOLD_KEV = 1.0  # Minimum energy for bremsstrahlung production
DELTA_RAY_THRESHOLD_KEV = 10.0  # Minimum energy for delta-ray production

# Density limits
MIN_DENSITY = 0.001  # Minimum physical density in g/cm³
MAX_DENSITY = 3.0  # Maximum physical density in g/cm³

# Binding energy formula constant
BINDING_ENERGY_CONSTANT = 0.0136  # K-shell binding energy constant (keV)

# Fluorescence yield constant
FLUORESCENCE_YIELD_CONSTANT = 1e6  # Fluorescence yield formula constant

# Characteristic X-ray energy constant
CHARACTERISTIC_ENERGY_CONSTANT = 0.0102  # K-alpha energy constant (keV)

# Radiation length formula constant
RADIATION_LENGTH_CONSTANT = 716.4  # Radiation length formula constant
RADIATION_LENGTH_LOG_CONSTANT = 287.0  # Radiation length log constant
