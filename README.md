# A Deep Multi-Agent Reinforcement Learning Approach to Solve Dynamic Job Shop Scheduling Problem
This repository includes the code of COR paper:

    @article{LIU2023106294,
    title = {A deep multi-agent reinforcement learning approach to solve dynamic job shop scheduling problem},
    journal = {Computers & Operations Research},
    volume = {159},
    pages = {106294},
    year = {2023},
    issn = {0305-0548},
    doi = {https://doi.org/10.1016/j.cor.2023.106294},
    url = {https://www.sciencedirect.com/science/article/pii/S0305054823001582},
    author = {Renke Liu and Rajesh Piplani and Carlos Toro}
    }

Please consider citing our paper if you found it helpful ^-^

## Repository Overview

This repo includes the code of simulation model, learning algorithm, and experimentation. Those files can be identified by their prefix:
> 1. "Asset": asset on shop floor, such as machines;
> 2. "Brain": learning algorithm for job sequencing;
> 3. "Event": dynamic events on shop floor, such as job arrival;
> 4. "Main": actual simulation processes, to train, or validate deep MARL-based algorithms;
> 5. "Static": experiments on static problem instances.

Data and trained parameters can be found in folders:
> 1. "experiment_result": results of experiments in dynamic environments;
> 2. "GA_static_validation": static problem instances and experiment result on static problems;
> 3. "trained_models": the trained parameters.

## User Guide

To use our code as the benchmark, kindly refer to "trained_models" folder for the trained parameters, and use the class "network_value_based" within "Brain_sequencing.py" file to build the neural network. The state and action functions can also be found in "Brain_sequencing.py" file.

An alternative way is to test your approach in our simulation model and context, you may create your algorithm and run the simulation in "Main_experiment.py" for comparison. Please refer to the comments in each module to see how to interact with the simulation model.
