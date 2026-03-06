# game-of-life-with-cnn
A rigorous exploration of cellular automata through the lens of deep learning, implementing a high-performance simulation of Conway's Game of Life and training CNNs to navigate the boundary between deterministic forward evolution and non-deterministic temporal reversal.

# Computational Ontogeny: Simulating and Predicting Conway’s Game of Life

This repository serves as a technical exploration into the emergent complexity of Cellular Automata. By leveraging high-performance Pythonic simulation techniques and Deep Learning architectures, this project investigates the predictability of systems governed by simple, local rules that result in complex, global behaviors.

## I. The Simulation Engine: Mathematical Foundations

The core of this project is a `Game` class designed for maximum computational efficiency. Unlike traditional iterative approaches that suffer from $O(N^2)$ complexity, this engine treats the grid as a digital signal to leverage vectorization.

### 1. Vectorized Convolution
To determine the survival of cells, the Moore neighborhood (8 surrounding cells) must be evaluated. This is achieved using a 2D convolution:
* **Kernel**: A 3x3 matrix with a zeroed center.
* **Operation**: The simulation utilizes `scipy.signal.convolve2d` to process the entire grid in a single pass, mapping localized rules to optimized C-level hardware instructions.



### 2. Toroidal Geometry
To simulate an infinite plane within a finite memory space, the grid is implemented as a **Torus**. By utilizing `boundary='wrap'`, cells on the far-right edge interact with those on the far-left, and the top interacts with the bottom. This removes boundary artifacts and ensures that energy and patterns are conserved within the system rather than being truncated.



---

## II. Neural Architecture: The God vs. The Devil

The primary objective of this study is to determine if a Convolutional Neural Network (CNN) can internalize the "physics" of a cellular automaton. This is split into two distinct epistemological challenges.

### 1. The "God" Model: Forward Determinism
The "God" model is tasked with predicting $G_{n+1}$ given $G_n$. 
* **The Logic**: Because the rules of the Game of Life are strictly deterministic, the transition from one state to the next is a surjective mapping. 
* **Performance**: The CNN acts as a universal function approximator. Given sufficient depth and filters, the model achieves near 100% accuracy, effectively internalizing the underlying mathematical laws of its environment.

### 2. The "Devil" Model: The Challenge of Inverse Entropy
The "Devil" model attempts the significantly more profound task of **Inverse Prediction**: determining $G_{n-1}$ from $G_n$.
* **Information Loss**: This is an inherently non-deterministic problem. In the Game of Life, multiple distinct parent configurations can lead to the exact same child configuration. This represents a "loss of information" over time.
* **The Complexity**: For any given dead cell, the model must deduce if it was previously dead, or if it died due to overpopulation or underpopulation. Because the mapping is not injective, the model must learn a probabilistic approximation.
* **Conclusion**: This highlights a fundamental boundary in Artificial Intelligence—performance is ultimately bounded by the inherent entropy and predictability of the system.



---

## III. Technical Specifications

* **Engine**: NumPy for multi-dimensional array manipulation and SciPy for signal processing.
* **Modeling**: TensorFlow/Keras utilizing 2D Convolutional layers with ReLU activation and Sigmoid output for binary state classification.
* **Hardware Optimization**: Designed for CPU-efficient simulation and GPU-accelerated model training.

## IV. Usage

```python
# Initialize a high-density initial state on a 60x120 toroidal field
system = Game(size = (60, 120), density = 0.35, seed = 2005)

# Calculate the next state using the vectorized engine
state = system.next(steps = 1)
