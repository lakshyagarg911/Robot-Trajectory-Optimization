# Robot Trajectory Optimization using Data & Learning

A machine learning approach for fast trajectory generation in a 2-DOF robotic manipulator.

Instead of solving a constrained optimization problem online for every motion request, this project first generates optimal trajectories using numerical optimization and then trains a neural network to directly predict smooth trajectories from start, goal, and waypoint constraints. The resulting model provides near-instant trajectory generation while preserving the characteristics of optimized solutions.

---

## Overview

Trajectory optimization produces smooth and dynamically feasible robot motions, but solving optimization problems online can be computationally expensive.

This project explores a learning-based alternative:

1. Generate optimal trajectories using constrained numerical optimization.
2. Build a dataset of optimized robot motions.
3. Train a neural network to imitate the optimizer.
4. Deploy the trained model through an interactive Streamlit dashboard.

The model learns to predict complete joint trajectories directly from:

* Start configuration
* Goal configuration
* Optional waypoint constraint

allowing trajectory generation in milliseconds instead of repeatedly solving optimization problems.

---

## Problem Formulation

A planar 2-link robotic arm must move between configurations while satisfying:

* Fixed start and end joint angles
* Zero velocity at start and end
* Optional waypoint constraints
* Smoothness requirements

The optimization objective minimizes trajectory acceleration while enforcing these constraints. Optimized trajectories are used as ground-truth labels for supervised learning.

---

## Dataset Generation

### Optimization-Based Data Generation

Training data is generated using GPU-accelerated trajectory optimization implemented in PyTorch.

For each sample:

* Random start configuration is generated
* Random goal configuration is generated
* Optional waypoint is sampled
* Optimization produces a smooth trajectory satisfying constraints

### Dataset Statistics

* **15,000 optimized trajectories**
* **50 timesteps per trajectory**
* **5 second motion duration**
* **50% trajectories contain waypoint constraints**
* **100 output values per sample**

  * 50 values for Joint 1
  * 50 values for Joint 2

Input format:

```text
[q1_start,
 q1_end,
 q2_start,
 q2_end,
 waypoint_time,
 waypoint_q1,
 waypoint_q2]
```

Output format:

```text
[q1(t0)...q1(t49),
 q2(t0)...q2(t49)]
```

---

## Neural Network Architecture

The optimizer is approximated using a Multi-Layer Perceptron (MLP).

### Architecture

```text
Input (7)
    ↓
Linear(256)
    ↓
Linear(512)
    ↓
Linear(512)
    ↓
Linear(256)
    ↓
Output (100)
```

Features:

* ReLU activations
* Batch normalization
* Dropout regularization
* AdamW optimization
* Learning-rate scheduling
* Early stopping

Total parameters: **~560,000**.

---

## Training Performance

The model converges smoothly and generalizes well to unseen trajectories.

### Training History

* Training MSE reduced from ~0.31 to ~0.085
* Validation MSE reduced to ~0.017
* Stable convergence with no severe overfitting

The validation loss consistently remains below the training loss because dropout is active during training and disabled during evaluation.

---

## Results

### Prediction Accuracy

Test-set performance:

* Overall trajectory RMSE ≈ **0.10 rad**
* Joint 1 error ≈ **0.12 rad**
* Joint 2 error ≈ **0.06 rad**

The predicted trajectories closely match numerically optimized solutions while maintaining smooth motion profiles.

### Qualitative Results

The model successfully learns:

* Smooth trajectory generation
* Waypoint satisfaction
* Start/end constraints
* Nonlinear trajectory shapes

Predicted trajectories closely follow the optimized trajectories across a wide range of robot configurations.

---

## Interactive Dashboard

A Streamlit application allows users to:

* Select start and end robot configurations
* Add waypoint constraints
* Compare optimization vs neural-network solutions
* Visualize joint trajectories
* View end-effector paths
* Animate robot motion
* Measure prediction error in real time

The dashboard demonstrates the trade-off between optimization accuracy and inference speed.

---

## Technology Stack

* Python
* PyTorch
* NumPy
* SciPy
* Streamlit
* Plotly
* Matplotlib

---

## Repository Structure

```text
├── data_generation.py
├── model_trainer.py
├── dashboard.py
├── trajectory_dataset.pkl
├── train_dataset.pkl
├── test_dataset.pkl
├── best_trajectory_model.pth
├── training_history.png
├── predictions.png
├── error_distribution.png
├── requirements.txt
└── README.md
```

---

## Future Work

* Higher DOF manipulators
* Obstacle avoidance constraints
* Transformer-based trajectory prediction
* Physics-informed neural networks
* Neural-network warm starts for optimization
* Real robot deployment

---

## Key Outcome

A 560K-parameter neural network was trained on 15,000 numerically optimized robot trajectories and learned to generate smooth, constraint-aware motions in real time, providing a practical approximation of computationally expensive trajectory optimization.
