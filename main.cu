//CS147 Final Project - Ricardo Villacana

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <chrono>

struct celestialObj {
    double x, y, z;
    double vx, vy, vz;
    double mass;
    double fx, fy, fz;
};

// Initializing each celestial object with a random position and mass, as well as zero velocity and force
void initializeObjects(std::vector<celestialObj>& celObjects, int n) {
    celObjects.resize(n);
    for (int i = 0; i < n; i++) {
        celObjects[i].x = rand() / (double)RAND_MAX;
        celObjects[i].y = rand() / (double)RAND_MAX;
        celObjects[i].z = rand() / (double)RAND_MAX;
        celObjects[i].vx = celObjects[i].vy = celObjects[i].vz = 0.0;
        celObjects[i].mass = rand() / (double)RAND_MAX;
        celObjects[i].fx = celObjects[i].fy = celObjects[i].fz = 0.0;
    }
}

// The CUDA kernel for calculating forces
__global__ void calculateForcesGPU(celestialObj* celObjects, int n, double G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute the index of the objects this thread is handling
    if (i < n) { // Ensure we don't go out of bounds
        celestialObj& p = celObjects[i];
        p.fx = p.fy = p.fz = 0.0; // Reset the forces
        for (int j = 0; j < n; j++) { // Loop over all other celestial objects in the cluster
            if (i != j) { // Don't calculate force with itself
                double dx = celObjects[j].x - p.x;
                double dy = celObjects[j].y - p.y;
                double dz = celObjects[j].z - p.z;
                double dist = sqrt(dx*dx + dy*dy + dz*dz);
                double F = G * p.mass * celObjects[j].mass / (dist*dist*dist);
                p.fx += F*dx;
                p.fy += F*dy;
                p.fz += F*dz;
            }
        }
    }
}

// Similar to the GPU calculateForcesGPU function, but runs on the CPU
// The forces are calculated in a pairwise manner
void calculateForcesCPU(std::vector<celestialObj>& celObjects, double G) {
    for (celestialObj& p : celObjects) {
        p.fx = p.fy = p.fz = 0.0;
    }
    for (int i = 0; i < celObjects.size(); i++) {
        for (int j = 0; j < celObjects.size(); j++) {
            if (i != j) { // Don't calculate force with itself
                double dx = celObjects[j].x - celObjects[i].x;
                double dy = celObjects[j].y - celObjects[i].y;
                double dz = celObjects[j].z - celObjects[i].z;
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                double F = G * celObjects[i].mass * celObjects[j].mass / (dist*dist*dist);
                celObjects[i].fx += F*dx;
                celObjects[i].fy += F*dy;
                celObjects[i].fz += F*dz;
            }
        }
    }
}

__global__ void moveObjectsGPU(celestialObj* celObjects, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute the index of the objects this thread is handling
    if (i < n) { // Ensure we don't go out of bounds
        celestialObj& p = celObjects[i];
        p.vx += p.fx / p.mass * dt;
        p.vy += p.fy / p.mass * dt;
        p.vz += p.fz / p.mass * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

// Update the position and velocity of each celestial object based on the forces acting on it
void moveObjects(std::vector<celestialObj>& celObjects, double dt) {
    for (celestialObj& p : celObjects) {
        p.vx += p.fx / p.mass * dt;
        p.vy += p.fy / p.mass * dt;
        p.vz += p.fz / p.mass * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

int main() {
    int n = 10000;
    int numSteps = 10;
    std::vector<celestialObj> celObjects(n);
    initializeObjects(celObjects, n); //Initializing the celestial objects celObjects
    double G = 6.67430e-11; // Gravitational constant
    double dt = 1e-5; // Time step
    std::vector<celestialObj> celObjectsCopy = celObjects;

    // Allocate memory on the GPU for the celestial objects
    celestialObj* d_celObjects;
    cudaMalloc(&d_celObjects, n * sizeof(celestialObj));

    // Calculate the number of CUDA blocks and threads
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int step = 0; step < numSteps; step++) {
        
        /*Comparing Force Caculations*/

        // Time the execution of the GPU version
        cudaMemcpy(d_celObjects, celObjects.data(), n * sizeof(celestialObj), cudaMemcpyHostToDevice);// Copy the celestial objects to the GPU
        // Start timing
        cudaEventRecord(start);
        // Call the CUDA kernel
        calculateForcesGPU<<<numBlocks, blockSize>>>(d_celObjects, n, G);
        // Stop timing
        cudaEventRecord(stop);
        // Copy the objects back to the CPU
        cudaMemcpy(celObjects.data(), d_celObjects, n * sizeof(celestialObj), cudaMemcpyDeviceToHost);
        // Make sure the event is finished before calculating the elapsed time
        cudaEventSynchronize(stop);
        // Calculate the elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GPU Force Calculation execution time: " << milliseconds << " ms" << std::endl;

        // Time the execution of the CPU version
        auto t1 = std::chrono::high_resolution_clock::now();// Record the start time
        // Run the CPU version
        calculateForcesCPU(celObjectsCopy, G);
        // Record the end time
        auto t2 = std::chrono::high_resolution_clock::now();
        // Calculate the elapsed time in milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "CPU Force Calculation execution time: " << duration / 1000.0 << " ms" << std::endl;

        //Verify that calculated forces from the GPU and CPU are the same
        bool resultsMatch = true;
        for (int i = 0; i < n; i++) {
            if (abs(celObjects[i].fx - celObjectsCopy[i].fx) > 1e-6
                || abs(celObjects[i].fy - celObjectsCopy[i].fy) > 1e-6
                || abs(celObjects[i].fz - celObjectsCopy[i].fz) > 1e-6) {
                resultsMatch = false;
                break;
            }
        }

        if (resultsMatch) {
            std::cout << "The force calculations from the GPU and CPU match!" << std::endl;
        } else {
            std::cout << "The force calculations from the GPU and CPU do NOT match!" << std::endl;
        }

        /*Comparing Move Objects Caculations*/

        // Time the execution of the GPU version
        cudaMemcpy(d_celObjects, celObjects.data(), n * sizeof(celestialObj), cudaMemcpyHostToDevice);// Copy the celestial objects to the GPU
        // Start timing
        cudaEventRecord(start);
        // Call the CUDA kernel
        moveObjectsGPU<<<numBlocks, blockSize>>>(d_celObjects, n, dt);
        // Stop timing
        cudaEventRecord(stop);
        // Copy the objects back to the CPU
        cudaMemcpy(celObjects.data(), d_celObjects, n * sizeof(celestialObj), cudaMemcpyDeviceToHost);
        // Make sure the event is finished before calculating the elapsed time
        cudaEventSynchronize(stop);
        // Calculate the elapsed time in milliseconds
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GPU Move Objects Calculation execution time: " << milliseconds << " ms" << std::endl;

        // Time the execution of the CPU version
        t1 = std::chrono::high_resolution_clock::now();// Record the start time
        // Run the CPU version
        moveObjects(celObjectsCopy, dt);
        // Record the end time
        t2 = std::chrono::high_resolution_clock::now();
        // Calculate the elapsed time in milliseconds
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "CPU Move Objects Calculation execution time: " << duration / 1000.0 << " ms" << std::endl;

        //Verify that calculated positions from the GPU and CPU are the same
        resultsMatch = true;
        for (int i = 0; i < n; i++) {
            if (abs(celObjects[i].x - celObjectsCopy[i].x) > 1e-6
                || abs(celObjects[i].y - celObjectsCopy[i].y) > 1e-6
                || abs(celObjects[i].z - celObjectsCopy[i].z) > 1e-6) {
                resultsMatch = false;
                break;
            }
        }

        if (resultsMatch) {
            std::cout << "The Move Objects calculations from the GPU and CPU match!" << std::endl;
        } else {
            std::cout << "The Move Objects calculations from the GPU and CPU do NOT match!" << std::endl;
        }

        std::cout << "Completed step " << step + 1<< "/" << numSteps << std::endl;

    }


    // Free GPU memory
    cudaFree(d_celObjects);
    // Destroy the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}