# Adaptive Grid Designs
This repository provides Python codes and data set for the paper ***Adaptive Grid Designs for Classifying Monotonic Binary Deterministic Computer Simulations***, submitted to JMLR.

## Dependencies
### Python Packages:

* rpy2
* skactiveml

### R Packages

* SLHD
* mined

### Other dependencies:

* active_learning.py
* partitioned_gp.py
* strategy_lib.py

These scripts are available on  [Lee et al. (2023)](https://github.com/cheolheil/ALIEN?tab=readme-ov-file "Title").



## Codes
### AdaptiveGridDesign.py
This script provides the main function of the paper, including the following:

* Python code for SG, SI, GG, GI, AG, AI, ALE, CAL, and AMC methods with detailed comments.
* Python code for OLH and MED methods with detailed comments.  These methods are generated from R package SLHD and mined. 
* Python code for PALC method with detailed comments. 
* Python code for simulations in Section 4.1, including the normal simulation and extreme cases

### crash.py
This script provides the results of Section 4.2, the road crash simulation:

* Python code for MC, AMC, SG, SI,ALE, GG, GI, AG, and AI methods suitable for ordinal variables
* Python code for road crash simulations in Section 4.2
* Needs crashData.csv


### Ice-breaking.py
This script provides the results of Section 5, the ice-breaking dynamic:

* Python code for ice-breaking simulations in Section 5
* Needs fgd.csv


### crashData.csv

  * Data set for road crash simulation with 44 crash occasions indexed by *caseID*
  * *eoff*: the off-road glance duration
  * *acc*: the decelaration
  * *crash*: the binray response
  * *caseID= 23, 25, 40, 41* contains only one kind of response, we skip them in simulation

### fgd.csv
  * Data set for ice-breaking simulation 
  * *v*: the initial velocity
  * *hnegative*: the additive inverse of ice thickness
  * *Enegative*: the additive inverse of elasticity modulus,
  * *Result*: binary response
  * All variables are re-scaled into [0,1]


