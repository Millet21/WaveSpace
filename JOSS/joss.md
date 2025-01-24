---
title: 'WaveSpace: analysis and simulation of cortical traveling waves'

tags:
  - Python
  - cortical traveling waves
  - simulation
  - phase-gradient
  - optical flow analysis
  - PCA 
  - circular-linear correlation
  
  
authors:
  - name: Kirsten Petras
    orcid: 0000-0001-5865-921X
    affiliation: 1
  - name: Dennis Croonenberg
  - name: Laura Dugué
    orcid: 0000-0003-3085-1458
    equal-contrib: false
    affiliation: "1, 2"

affiliations:
  - name: Université Paris Cité, INCC UMR 8002, CNRS, F-75006 Paris, France
    index: 1
  - name: Institut Universitaire de France (IUF), Paris, France
    index: 2


date: 15 January 2025
bibliography: references.bib

---


# Statement of need

Oscillatory cortical activity has been found to systematically propagate across space [@Zhang]. Various approaches to detect and characterize such spatiotemporal patterns of activity, often referred to as "oscillatory cortical traveling waves (ocTW)," have emerged in the literature. Typically, laboratories develop customized pipelines tailored to their experimental requirements and software platform preferences [@Alexander;@Muller;@Alamia;@Das] but see also [@Gutzen] for a notable exception.

The diversity of methods and implementations found in the literature poses challenges for researchers, both in selecting the one most suitable for their own studies and in directly comparing the performance of different pipelines. WaveSpace addresses this gap by integrating commonly used strategies into a single modular framework. This framework ensures that modules for preprocessing, data decomposition, spatial arrangement of sensor positions, wave analysis, and evaluation are interchangeable and adhere to a consistent logic. The resulting pipelines are ready-to-use in empirical studies [@Petras;@Fakche]   

# Functionality
WaveSpace contains 5 modules:

- Decomposition: Provides multiple techniques to decompose broadband data into frequency components, including FFT-based methods (e.g., wavelets, filter-Hilbert), empirical mode decomposition (EMD), and generalized phase analysis.

- Spatial Arrangement: Includes methods to map 3D sensor positions onto 2D regular grids using approaches including multidimensional scaling (MDS) and isomap, including interpolation options.

- Wave Analysis: Offers a variety of analysis methods, such as 2D FFT, optical flow analysis, phase gradient methods, and principal component analysis (PCA).

- Simulation: Functions to simulate traveling and standing waves with both linear and nonlinear properties, as well as incorporating noise.

- Plotting: Contains visualization tools for each analysis option.

The entire framework is comprehensively documented and includes example scripts to facilitate its adoption.

![WaveSpace Module Overview](WaveSpace_overview.png)
*Figure 1: Overview of WaveSpace modules.*

# Funding

This project received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No. 852139 - Laura Dugué).

# Toolbox dependencies

Below are the direct dependencies of WaveSpace. Dependencies of dependencies, as well as requried versions can be found in the environment file.


# References
