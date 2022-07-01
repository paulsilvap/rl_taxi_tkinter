# Simulation Environment for Electric Taxi Management
This environment uses Tkinter to render the GUI and it is meant for testing different reinforcement learning algorithms.

## Installation
First, make sure that you have the right tkinter version that goes with the python version you are going to use. And then install the requirements with pip:

```bash
pip install -r requirements.txt
```

## Implemented Algorithms

- DQN (closely follows the DQN implementation from this amazing [repository](https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code))

## Environments
- Grid Environment: 
  * environment:
    + Passengers = 1 
    + Charging Stations = 1
    + E-taxi: 1 
  * multi_env:
    + Passengers = 5 
    + Charging Stations = 4
    + E-taxi: 1 
- Graph Environments:
  * dispatch_env:
    * Variable Number of passengers
    * Charging Stations = 2
    * E-taxi: 1
    * Better control over parameters of EV
    * Real Network topology, traffic conditions and charging station locations

## Example
Run min_run.py to see a preview of the Graph Environment for a single episode. This script has most of the information necessary to run the environment.



