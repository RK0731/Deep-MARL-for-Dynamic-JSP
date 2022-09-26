# A Deep Multi-Agent Reinforcement Learning Approach to Solve Dynamic Job Shop Scheduling Problem
This repository includes the code used in the paper submission: 

**Liu, R.**, Piplani, R., & Toro, C. (2022). A Deep Multi-Agent Reinforcement Learning Approach to Solve Dynamic Job Shop Scheduling Problem. Computers and Operations Research.

## Repository Overview

This repo includes the simulation model, learning algorithm, and experiment codes. Those files can be identified by their prefix:
> 1. "Asset": asset on shop floor, such as machines;
> 2. "Brain": learning algorithm for job sequencing;
> 3. "Event": dynamic events on shop floor, such as job arrival;
> 4. "Main": actual simulation processes, to train, or validate deep MARL-based algorithms;
> 5. "Static": experiments on static problem instances.

Data and trained parameters can be found in folders:
> 1. "experiment_result": results of experiments in dynamic envrioenments;
> 2. "GA_static_validation": static problem instances and experiment result on static problems;
> 3. "trained_models": the trained parameters.