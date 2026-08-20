# Chem Analysis

Your one-stop shop for analyzing chemistry data.

This package was developed general, but special attention was paid to analyze large data sets 
(example: analyzing 100 NMR at once). 

Design Philosophy:
* Handle large data sets 
  * Support analyzing 1000s of NMR at once. (like those generated from kinetic analysis)
  * visualization is also built to support this.
* Modular 
  * be able to turn on or off or switch out methods with minimal code change
* Explict
  * Alot of analytic software automatically performs data transforms and hides this from the user making it hard to 
  truly know what's going on with your data. Here everything needs to be called explicitly, but typical processing
  steps are suggested in several of the examples.


**Support data types**:
* IR
* NMR (Bruker, Spinsolve) - 1D only
* HPLC
* SEC (GPC)
* GC
* UV-Vis
* Mass Spec.

## Installation
[pypi page](https://pypi.org/project/chem-analysis/)

`pip install chem_analysis`

## Capabilities
### Processing Methods:
* Baseline correction
* Peak Picking
* Phase correction (NMR)
* Referencing (NMR)


### Analysis Methods:
* Integration
* Peak fitting
* Multi-component analysis (MCA)

## Plotting / GUI



## Examples
Examples of processing data are found under /dev/data_workups.

## Contributing

Contributions are welcomed! Best practice is to open an issue with your idea, and I will let you know if it
is a good fit for the project. If you are interested in helping code the addition please mention that as well. 

## Publication Data
Data for "Leveraging Automation and Inline and Online Monitoring for Data-Rich Experimentation of Photopolymerization in Segmented Flow" is provided, as well as scrips for analyzing and visualizing this data and training gaussian process regression models. The trained models are also provided. 
