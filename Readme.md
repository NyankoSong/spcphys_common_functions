# spcphys-common-functions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A collection of commonly used Python functions for my space physics research.

## Features

### Parameters (`spcphys_common_functions.parameters`)
- **Alfvénic Parameters** (`alfvenic_parameters`): Calculate cross-helicity, residual energy, Alfvén ratio, compressibility, and correlation coefficients. Supports both single-window and time-series analysis with optional multiprocessing.
- **Plasma Beta** (`plasma_beta`): Compute thermal pressure, magnetic pressure, and plasma beta.
- **Thermal Velocity/Energy/Temperature** (`vth_E_T`): Conversions between thermal velocity (vth), energy (E), and temperature (T). Also includes temperature tensor decomposition into parallel and perpendicular components.
- **Coulomb Collisional Age** (`coulomb_collisional_age`): Calculate collisional age following Tracy et al. (2015), supporting arbitrary test and field particle species.
- **Effect Size** (`effectsize`): Cohen's d (Hedges' g) effect size calculations with confidence intervals, supporting log-scale and rank-based transformations.
- **Minimum Variance Analysis** (`minimum_variance`): MVA for vector data analysis, returning rotated data, eigenvectors, and eigenvalues.

### Processing (`spcphys_common_functions.processing`)
- **Backmapping** (`backmapping`): Ballistic backmapping for solar wind source region identification. Includes single-point backmapping and dual-spacecraft observation difference calculations with multiprocessing support.
- **CDF Processing** (`cdf_process`): Read and batch-process CDF files from spacecraft missions. Automatically generate info CSV files for CDF datasets.
- **Network Tools** (`net_tools`): Asynchronously fetch CDF files from CDAWeb with retry logic and concurrent download support.
- **Time Window** (`time_window`): Sliding time window analysis tools for creating windows based on time duration or element count.
- **Preprocessing** (`preprocess`): Data cleaning, interpolation (including angular interpolation for degrees), down-sampling, NaN handling, and datetime conversions.
- **VDF Processing** (`vdf_process`): Velocity Distribution Function analysis including coordinate transformation to field-aligned systems, spherical to Cartesian conversion, 3D grid interpolation, and 1D/2D visualization.
- **Vector Transformations** (`vec_cart_sph`): Cartesian-Spherical coordinate conversions and quaternion-based vector rotations.
- **Plot Tools** (`plot_tools`): Comprehensive visualization utilities including 1D/2D/3D histograms, box/violin statistics, time-series mesh plots, auto-downsampling for large datasets, and customizable error bar plotting.

### Utilities (`spcphys_common_functions.utils`)
- Helper functions for parallel processing (CPU core allocation)

### Dependencies

- Python >= 3.11
- numpy >= 2.3.0
- scipy >= 1.15.3
- astropy >= 7.1.0
- sunpy >= 6.1.1
- pandas >= 2.3.0
- matplotlib >= 3.10.3
- cdflib >= 1.3.4
- cdasws >= 1.8.11
- aiohttp >= 3.12.8
- tqdm >= 4.67.1
- beartype >= 0.21.0
- nest_asyncio >= 1.6.0

## Module Structure

```
spcphys_common_functions/
├── parameters/
│   ├── alfvenic_parameters.py  # Alfvén wave analysis
│   ├── plasma_beta.py          # Plasma beta calculations
│   ├── vth_E_T.py              # Thermal velocity/energy/temperature
│   ├── coulomb_collisional_age.py  # Collisional age
│   ├── effectsize.py           # Statistical effect size
│   └── minimum_variance.py     # MVA analysis
├── processing/
│   ├── backmapping.py          # Solar wind backmapping
│   ├── cdf_process.py          # CDF file processing
│   ├── net_tools.py            # Network/download tools
│   ├── preprocess.py           # Data preprocessing
│   ├── time_window.py          # Time window utilities
│   ├── vdf_process.py          # VDF analysis
│   ├── vec_cart_sph.py         # Coordinate transformations
│   └── plot_tools.py           # Plotting utilities
└── utils/
    └── utils.py                # General utilities
```

## References

This package implements methods from the following publications:

- Yao et al. (2013) - Small-scale pressure-balanced structures. [ApJ 776:94](https://doi.org/10.1088/0004-637X/776/2/94)
- Wu et al. (2021) - Magnetic and velocity fluctuations in the near-Sun region. [ApJ 922:92](https://doi.org/10.3847/1538-4357/ac3331)
- Tracy et al. (2015) - Thermalization of heavy ions in the solar wind. [ApJ 812:170](https://doi.org/10.1088/0004-637X/812/2/170)
- Bale et al. (2019) - Highly structured slow solar wind. [Nature 576:237-242](https://doi.org/10.1038/s41586-019-1818-7)
- Owen et al. (2020) - Solar Orbiter SWA suite. [A&A 642:A16](https://doi.org/10.1051/0004-6361/201937259)
- De Marco et al. (2023) - Separating proton populations in VDFs. [A&A 669:A108](https://doi.org/10.1051/0004-6361/202243719)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.