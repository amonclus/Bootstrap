# Network risk analysis tool
This repository contains a network risk analysis tool that helps identify potential 
vulnerabilities of the network structure in relation to cascade effects. It uses an algorithm based on
bootstrap percolation to simulate the spread of failures through a network and identify critical nodes 
that could lead to a widespread failure.

## Installation
To install the necessary dependencies, run the following command from the root directory of the project:
```bash
pip install -r src/requirements.txt
```

## Usage
To run the network risk analysis tool, use the following command:
```
bash python src/Main.py 
```

The system will then give the user three options:
1. Generate an artificial graph:   
    - Erdos-Renyi graph (Number of nodes and edge probability)
    - Random geometric graph (Number of nodes and radius)
    - Lattice graph (Size of the lattice)
   
2. Load a graph from a file:
   - The file should be in the format of DIMACS, an edge list or GML.
   
3. Run a sweep of the parameter space on all graph types:
    - This will run the algorithm on a range of parameters for all graph 
   types and display all results. This serves as an experimental tool
    to understand the behavior of the algorithm under different conditions.

After loading or generating a graph, the user will be prompted to select the parameters 
for the bootstrap percolation algorithm, such as the threshold for node failure and the initial seed probability.

### Results
The results of the analysis will be displayed in the console, showing a single trial and its output in the form of 
an animation of the cascade process and the metrics obtained over multiple trials of the same graph in order
to minimeze the effect of randomness. The following metrics are measured:
- Final cascade size: The proportion of nodes that failed at the end of the cascade process.
- Time to cascade: The number of iterations it takes for the cascade to stabilize 
- Critical seed size: The minimum initial seed size required to trigger a cascade that affects a significant portion of the network.
- Cascade threshold: The minimum threshold value that leads to a cascade affecting a significant portion of the network.
- Cascade probability: The probability that a cascade occurs given a certain initial seed size and threshold.

From these metrics a final robustness score is calculated, which is a composite measure of the network's vulnerability to cascade effects.

## UI
To launch the interactive user interface, run the following command:
```
bash streamlit run src/app.py
```