# Workspace Context

## Project Overview
* the project is a deep learning project.
* the data is time series data, stores at ./dataset
* the tasks are prediction or classification

### Dataset and dataloader

* Dataset and dataloader are implemented at ./data_provider
* ./data_provider/data_factory.py is the dispatcher, def data_provider(args, flag) returns the dataset and dataloader
* ./data_provider/data_loader.py implements the specific Dataset 

### Model
* models are implemented at ./models using modules at ./layers

### Train, valid, test
* Train, valid, test are implemented at ./exp

### Define and Launch an experiment 
* the bash scripts used to run an experiment are defined at ./scripts 
* ./run.py is the top-level program that defined configuratio parameters for an experiment



