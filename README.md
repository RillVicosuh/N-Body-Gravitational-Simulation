# Accelerating the N-Body Gravitational Problem Using CUDA

![Project Status: Completed](https://img.shields.io/badge/Project%20Status-Completed-green)

This project provides an optimized solution for the classical N-Body gravitational calculation problem, a computationally intensive task that predicts the individual motions of a group of celestial objects interacting with each other gravitationally. The project leverages the parallel processing capabilities of CUDA and GPUs to transform a CPU-based sequential computation into an efficient GPU-accelerated parallel computation.

## Overview

In the traditional N-Body problem, each celestial body's position is calculated sequentially, leading to a time complexity of O(n^2). This project transforms this process, allowing for simultaneous calculations of the gravitational forces acting upon an object and the new positions of multiple celestial bodies, which significantly reduces computation time. The GPU's parallel processing capabilities are leveraged fully by partitioning the problem space into threads and thread blocks, with each body's forces and new position calculated by one thread.

### CPU vs. GPU Implementation

The CPU-based sequential implementation and the GPU-accelerated parallel implementation are both covered in this project. In the CPU implementation, the calculations for each object are performed in sequence, with each object waiting for the previous one to complete before starting its calculation. On the other hand, the GPU implementation performs many calculations simultaneously, significantly reducing computation time. 

### Results

Force calculations executed on the GPU showed consistent performance, averaging around 23.1 ms, approximately 32.7 times faster than the CPU implementation. Particle movement computations on the GPU averaged about 0.012632 ms, approximately six times faster than the CPU.

## How to Run the Code

1. Ensure you have the CUDA toolkit installed.
2. Use the command `make` to compile the program, assuming you have the makefile.
3. Run the program with the command `./main`. The necessary components are randomly generated, and the computation will start.

## Limitations and Future Work

While the project successfully optimized the N-Body gravitational calculator using the GPU, efforts to implement shared memory for further optimization encountered obstacles. For reasons not entirely determined, incorporating shared memory led to the results from the GPU calculations zeroing out. Despite significant troubleshooting, the exact cause of this issue could not be identified. Future work could explore this problem further for potential improvements in the program's performance.

## Demo

[Video Link](https://youtu.be/PHWbimjlQvM) 
