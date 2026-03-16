# Neural Networks as Graphs: Structure, Dynamics, and Resilience

This repository contains the Python source code, research paper, and presentation for my university seminar project at the Faculty of Electrical Engineering, University of Sarajevo. 

The project focuses on modeling neural networks as directed graphs, analyzing their structural resilience, and simulating signal dynamics using graph theory concepts.

## 📂 Repository Structure

* `src/` - Python source code for graph generation, simulation, and resilience analysis.
* `docs/` - The complete seminar paper (PDF) detailing the theoretical background and methodology.
* `presentation/` - Slide deck used for the project defense.

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Libraries:** NetworkX, NumPy, Matplotlib (for network visualization and animation)
* **Concepts:** Graph Theory, Barabási-Albert Model, Dale's Law, Centrality Metrics

## 🔍 Key Features of the Project

1. **Structural Modeling:** Generating weighted, directed graphs that simulate biological synapses, incorporating Dale's Law for excitatory and inhibitory neurons.
2. **Resilience Analysis:** Testing network stability against targeted node removals using Betweenness Centrality metrics.
3. **Signal Dynamics Simulation:** Tracking signal propagation through the network from core nodes to the periphery (and vice versa) with animated visualizations.

## 🚀 How to Run the Code

To run the analysis locally, clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/rubina-rekic/neural-networks-graph-analysis.git

# Install dependencies (ensure you have Python installed)
pip install networkx numpy matplotlib

# Run the main script (adjust the filename if needed)
python src/main.py
