# cubewalkers-analysis
This repository contains all analysis result published on 

> Park KH, Costa FX, Rocha LM, Albert R, Rozum JC. Models of Cell Processes are Far from the Edge of Chaos. PRX Life 1, 023009 (2023)

which can be accessed here: https://doi.org/10.1103/PRXLife.1.023009.

## Documentation

### models
Here we have the original models from the Cell Collective (https://cellcollective.org), and the models we corrected. The corrections are made to better resemble the model in the original publications. For example, some models are only valid in a particular cellular context. See the list of alterations at `corrected_models_list.csv`.

### analysis
Files in this directory cacalculates various dynamic parameters of Boolean models, such as the Derrida coefficient or the converged average node values.
Requires `cubewalkers`, `cupy` and GPU supporting CUDA.

### notebooks
We have notebooks that generate figures and tables in the paper, either by processing the data in the analysis folder or by running simple analyses directly.  
`perturbation_output_plots` Figures 1, 4-6, 14-19  
`node_average_values_differences_plots` Figure 3  
`simple_benchmark_plots` Figures 7, 8  
`convergence_of_averages` Figure 9  
`analysis_of_specific_models` Some results are used to generate Figures 10 and 11  
`source_nodes_plots` Figure 12  
`correction_comparison_plots` Figure 13  
`models_description_tables` Table II-XIV  

### data
All data (except data generated during the analysis phase), figures, and tables generated are stored here.

## Requirements

`cubewalkers (v1.3.7+)` https://github.com/jcrozum/cubewalkers

`cupy (v12.2.0+)` https://cupy.dev/ <br>
Note: `cupy` requires cuda11x or other version depending on installed CUDA version  

`pyboolnet (v3.0.9+)` https://github.com/hklarner/pyboolnet <br>
Note 1: `pyboolnet` requires `pyeda`, which can be difficult to install in Windows; it is recommended to obtain a `pyeda` Windows wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyeda <br>
Note 2: `pyboolnet` also requires `clasp` and `gringo` to be installed separately on Linux systems  

`pystablemotifs (v3.0.4+)` https://github.com/jcrozum/pystablemotifs

`cana (v0.2.0+)` https://github.com/CASCI-lab/CANA

`pandas (v2.0.1+)` https://pandas.pydata.org/

`numy (v1.19.2+)` https://numpy.org/

`scipy (v1.10.1+)` https://scipy.org/

`matplotlib (v3.2.1+)` https://matplotlib.org/

`jinja2 (v3.1.2+)` https://jinja.palletsprojects.com/en/2.10.x/

`setuptools (v65.5.0+)` https://github.com/pypa/setuptools
