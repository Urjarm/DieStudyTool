# DieStudyTool

## Requirements
```text
python: 3.11
```

## Installation
- Download this directory to your local machine
- Create a new virtual environment in the main folder of this directory:
```bash
python -3.11 -m venv .venv
```
- Activate the virtual environment:
```bash
.venv/bin/activate
```
- Install the packages into the virtual environment, which this program requires to run:
```bash
pip install -r requirements.txt
```

## Run
In order to run this program, the "DieStudyTool.ipynb" has to be executed. Before this can be done, you should enter the important variables in the "variables.py" file. These are mainly "images_directory" and "results_directory", as these paths are dependent on your local storaging.

**(Optional)**: The variables "matching_computation_method" and "distance_computation_method" can be used to change the function that is used for computing the solution. The preselected functions proved to be the best in (CITATION). "number_of_clusters" can be used to change the number of clusters that the tool outputs. The two remaining variables "matching_file_name" and "clustering_file_name" merely change the name of the output files containing the results.

## Affiliation
This tool was developed as part of a thesis for the Big Data Lab of the Goethe Universität in Frankfurt am Main.
http://www.bigdata.uni-frankfurt.de/
https://github.com/Frankfurt-BigDataLab