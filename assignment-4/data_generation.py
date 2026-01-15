import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time

# ---------------- GPU Setup ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA cores: {torch.cuda.get_device_properties(0).multi_processor_count * 128}")  # Approximate
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Using device: {device}")
print("=" * 60)
print()

# ---------------- Parameters ----------------
T = 5.0  # Total time
N = 50  # Number of time steps
dt = T / (N - 1)
t_np = np.linspace(0, T, N)
t = torch.linspace(0, T, N, device=device)


# ---------------- PyTorch Trajectory Optimization with Waypoints ----------------
def quartic_trajectory_torch(q_start, q_end, T, t):
    """Generate initial quartic trajectory using PyTorch"""
    a0 = q_start
    a1 = 0
    a2 = 0

    # Solve for a3, a4
    A = torch.tensor([
        [T ** 3, T ** 4],
        [3 * T ** 2, 4 * T ** 3]
    ], dtype=torch.float32, device=t.device)
    b = torch.tensor([
        q_end - q_start,
        0
    ], dtype=torch.float32, device=t.device)

    a3, a4 = torch.linalg.solve(A, b)

    q = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4
    return q


class TrajectoryOptimizer(nn.Module):
    """PyTorch-based trajectory optimizer with waypoint constraints"""

    def __init__(self, N, dt, device='cuda'):
        super().__init__()
        self.N = N
        self.dt = dt
        self.device = device

    def acceleration_cost(self, q1, q2):
        """Compute acceleration cost for trajectories"""
        # Finite difference approximation of second derivative
        q1_dd = (q1[..., 2:] - 2 * q1[..., 1:-1] + q1[..., :-2]) / (self.dt ** 2)
        q2_dd = (q2[..., 2:] - 2 * q2[..., 1:-1] + q2[..., :-2]) / (self.dt ** 2)

        cost = torch.sum(q1_dd ** 2 + q2_dd ** 2, dim=-1)
        return cost

    def velocity_penalty(self, q, q_start_vel=0.0, q_end_vel=0.0, weight=1000.0):
        """Penalize non-zero velocities at start and end"""
        v_start = (q[..., 1] - q[..., 0]) / self.dt
        v_end = (q[..., -1] - q[..., -2]) / self.dt

        penalty = weight * (v_start ** 2 + v_end ** 2)
        return penalty

    def position_penalty(self, q, q_start, q_end, weight=10000.0):
        """Penalize deviation from start/end positions"""
        penalty = weight * ((q[..., 0] - q_start) ** 2 + (q[..., -1] - q_end) ** 2)
        return penalty

    def waypoint_penalty(self, q1, q2, waypoint_time_idx, waypoint_q1, waypoint_q2, weight=5000.0):
        """Penalize deviation from waypoint at specified time index"""
        if waypoint_time_idx is None:
            return 0.0

        q1_at_waypoint = q1[..., waypoint_time_idx]
        q2_at_waypoint = q2[..., waypoint_time_idx]

        penalty = weight * ((q1_at_waypoint - waypoint_q1) ** 2 + (q2_at_waypoint - waypoint_q2) ** 2)
        return penalty

    def total_cost(self, q1, q2, q1_start, q1_end, q2_start, q2_end,
                   waypoint_time_idx=None, waypoint_q1=None, waypoint_q2=None):
        """Combined cost function with constraints as penalties"""
        acc_cost = self.acceleration_cost(q1, q2)

        # Soft constraints (penalties)
        pos_penalty = (
                self.position_penalty(q1, q1_start, q1_end) +
                self.position_penalty(q2, q2_start, q2_end)
        )

        vel_penalty = (
                self.velocity_penalty(q1) +
                self.velocity_penalty(q2)
        )

        # Waypoint penalty
        wp_penalty = self.waypoint_penalty(q1, q2, waypoint_time_idx, waypoint_q1, waypoint_q2)

        return acc_cost + pos_penalty + vel_penalty + wp_penalty


def optimize_trajectories_batch(q1_starts, q1_ends, q2_starts, q2_ends,
                                waypoint_time_indices, waypoint_q1s, waypoint_q2s,
                                N, dt, T, t, device,
                                num_iterations=100, lr=0.05):
    """
    Optimize multiple trajectories in parallel using PyTorch with waypoint constraints

    Args:
        q1_starts, q1_ends, q2_starts, q2_ends: Tensors of shape (batch_size,)
        waypoint_time_indices: Tensor of shape (batch_size,) - time index for waypoint
        waypoint_q1s, waypoint_q2s: Tensors of shape (batch_size,) - waypoint joint angles
        N, dt, T, t: Time parameters
        device: torch device
        num_iterations: Number of optimization steps
        lr: Learning rate

    Returns:
        q1_opt, q2_opt: Optimized trajectories of shape (batch_size, N)
    """
    batch_size = q1_starts.shape[0]

    # Initialize with quartic trajectories
    q1_init = torch.stack([
        quartic_trajectory_torch(q1_starts[i], q1_ends[i], T, t)
        for i in range(batch_size)
    ])

    q2_init = torch.stack([
        quartic_trajectory_torch(q2_starts[i], q2_ends[i], T, t)
        for i in range(batch_size)
    ])

    # Make trajectories learnable parameters (but keep endpoints fixed)
    q1_middle = nn.Parameter(q1_init[:, 1:-1].clone())
    q2_middle = nn.Parameter(q2_init[:, 1:-1].clone())

    optimizer_module = TrajectoryOptimizer(N, dt, device)
    optimizer = torch.optim.Adam([q1_middle, q2_middle], lr=lr)

    # Optimize with progress tracking
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Reconstruct full trajectories with fixed endpoints
        q1_full = torch.cat([
            q1_starts.unsqueeze(1),
            q1_middle,
            q1_ends.unsqueeze(1)
        ], dim=1)

        q2_full = torch.cat([
            q2_starts.unsqueeze(1),
            q2_middle,
            q2_ends.unsqueeze(1)
        ], dim=1)

        # Vectorized cost computation for entire batch
        acc_cost = optimizer_module.acceleration_cost(q1_full, q2_full)
        pos_penalty = (
                optimizer_module.position_penalty(q1_full, q1_starts, q1_ends) +
                optimizer_module.position_penalty(q2_full, q2_starts, q2_ends)
        )
        vel_penalty = (
                optimizer_module.velocity_penalty(q1_full) +
                optimizer_module.velocity_penalty(q2_full)
        )

        # Waypoint penalty (vectorized for batch)
        wp_penalty = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            if waypoint_time_indices[i] >= 0:
                wp_idx = waypoint_time_indices[i].item()
                wp_penalty[i] = 5000.0 * (
                        (q1_full[i, wp_idx] - waypoint_q1s[i]) ** 2 +
                        (q2_full[i, wp_idx] - waypoint_q2s[i]) ** 2
                )

        total_cost = (acc_cost + pos_penalty + vel_penalty + wp_penalty).mean()

        # Backpropagation
        total_cost.backward()
        optimizer.step()

        # Early stopping
        if total_cost.item() < best_loss:
            best_loss = total_cost.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final trajectories
    with torch.no_grad():
        q1_opt = torch.cat([
            q1_starts.unsqueeze(1),
            q1_middle,
            q1_ends.unsqueeze(1)
        ], dim=1)

        q2_opt = torch.cat([
            q2_starts.unsqueeze(1),
            q2_middle,
            q2_ends.unsqueeze(1)
        ], dim=1)

    return q1_opt, q2_opt


# ---------------- Dataset Generation ----------------
def generate_dataset_pytorch(num_samples=1000, batch_size=128, save_path='trajectory_dataset.pkl',
                             waypoint_probability=0.5):
    """
    Generate dataset using PyTorch GPU-accelerated optimization with waypoints

    Args:
        num_samples: Total number of trajectories to generate
        batch_size: Number of trajectories to optimize in parallel
        save_path: Path to save the dataset
        waypoint_probability: Probability of including a waypoint (0.0 to 1.0)

    Returns:
        Dictionary containing inputs, outputs, and metadata
    """
    # Define ranges
    q1_range = (-np.pi, np.pi)
    q2_range = (-np.pi / 2, np.pi / 2)

    all_inputs = []
    all_outputs = []

    print(f"Generating {num_samples} optimized trajectories with waypoints...")
    print(f"Waypoint probability: {waypoint_probability * 100:.0f}%")
    print(f"Batch size: {batch_size} (trajectories optimized in parallel)")
    print(f"Number of batches: {(num_samples + batch_size - 1) // batch_size}")
    print()

    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Determine actual batch size for this iteration
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # Sample random start and end configurations
        q1_starts = torch.FloatTensor(current_batch_size).uniform_(*q1_range).to(device)
        q1_ends = torch.FloatTensor(current_batch_size).uniform_(*q1_range).to(device)
        q2_starts = torch.FloatTensor(current_batch_size).uniform_(*q2_range).to(device)
        q2_ends = torch.FloatTensor(current_batch_size).uniform_(*q2_range).to(device)

        # Sample waypoints
        waypoint_time_indices = []
        waypoint_q1s = []
        waypoint_q2s = []
        has_waypoint = []

        for i in range(current_batch_size):
            if np.random.rand() < waypoint_probability:
                # Add waypoint at random time between 20% and 80% of trajectory
                wp_time_idx = np.random.randint(int(N * 0.2), int(N * 0.8))
                wp_q1 = np.random.uniform(*q1_range)
                wp_q2 = np.random.uniform(*q2_range)

                waypoint_time_indices.append(wp_time_idx)
                waypoint_q1s.append(wp_q1)
                waypoint_q2s.append(wp_q2)
                has_waypoint.append(True)
            else:
                # No waypoint
                waypoint_time_indices.append(-1)  # -1 indicates no waypoint
                waypoint_q1s.append(0.0)
                waypoint_q2s.append(0.0)
                has_waypoint.append(False)

        waypoint_time_indices = torch.tensor(waypoint_time_indices, device=device)
        waypoint_q1s = torch.tensor(waypoint_q1s, device=device)
        waypoint_q2s = torch.tensor(waypoint_q2s, device=device)

        # Optimize trajectories in parallel
        q1_opt, q2_opt = optimize_trajectories_batch(
            q1_starts, q1_ends, q2_starts, q2_ends,
            waypoint_time_indices, waypoint_q1s, waypoint_q2s,
            N, dt, T, t, device
        )

        # Move to CPU and convert to numpy
        q1_opt_np = q1_opt.cpu().numpy()
        q2_opt_np = q2_opt.cpu().numpy()
        q1_starts_np = q1_starts.cpu().numpy()
        q1_ends_np = q1_ends.cpu().numpy()
        q2_starts_np = q2_starts.cpu().numpy()
        q2_ends_np = q2_ends.cpu().numpy()
        waypoint_time_indices_np = waypoint_time_indices.cpu().numpy()
        waypoint_q1s_np = waypoint_q1s.cpu().numpy()
        waypoint_q2s_np = waypoint_q2s.cpu().numpy()

        # Store results
        for i in range(current_batch_size):
            if has_waypoint[i]:
                # Input: [q1_start, q1_end, q2_start, q2_end, waypoint_time, waypoint_q1, waypoint_q2]
                input_vec = np.array([
                    q1_starts_np[i], q1_ends_np[i],
                    q2_starts_np[i], q2_ends_np[i],
                    waypoint_time_indices_np[i] * dt,  # Convert index to time
                    waypoint_q1s_np[i], waypoint_q2s_np[i]
                ])
            else:
                # No waypoint: use -1 as indicator
                input_vec = np.array([
                    q1_starts_np[i], q1_ends_np[i],
                    q2_starts_np[i], q2_ends_np[i],
                    -1.0, 0.0, 0.0  # -1 indicates no waypoint
                ])

            output_vec = np.hstack([q1_opt_np[i], q2_opt_np[i]])

            all_inputs.append(input_vec)
            all_outputs.append(output_vec)

    # Convert to numpy arrays
    inputs = np.array(all_inputs)
    outputs = np.array(all_outputs)

    # Count trajectories with waypoints
    num_with_waypoints = np.sum(inputs[:, 4] >= 0)

    print(f"\nDataset generation complete!")
    print(f"Total trajectories: {len(inputs)}")
    print(f"Trajectories with waypoints: {num_with_waypoints}")
    print(f"Trajectories without waypoints: {len(inputs) - num_with_waypoints}")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")

    # Create dataset dictionary
    dataset = {
        'inputs': inputs,
        'outputs': outputs,
        'metadata': {
            'num_samples': len(inputs),
            'N': N,
            'T': T,
            'dt': dt,
            'q1_range': q1_range,
            'q2_range': q2_range,
            'waypoint_probability': waypoint_probability,
            'description': 'PyTorch GPU-optimized joint trajectories with waypoints for 2-link planar robot',
            'input_format': '[q1_start, q1_end, q2_start, q2_end, waypoint_time, waypoint_q1, waypoint_q2]',
            'waypoint_note': 'waypoint_time = -1 indicates no waypoint',
            'generated_with_pytorch': True,
            'device': str(device)
        }
    }

    # Save dataset
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {save_path}")

    return dataset


def visualize_samples(dataset, num_samples=4):
    """Visualize sample trajectories with waypoints"""
    inputs = dataset['inputs']
    outputs = dataset['outputs']
    N = dataset['metadata']['N']
    T = dataset['metadata']['T']
    dt = dataset['metadata']['dt']
    t = np.linspace(0, T, N)

    indices = np.random.choice(len(inputs), size=min(num_samples, len(inputs)), replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(indices):
        input_vec = inputs[sample_idx]
        output_vec = outputs[sample_idx]

        q1_start, q1_end, q2_start, q2_end, wp_time, wp_q1, wp_q2 = input_vec
        q1_traj = output_vec[:N]
        q2_traj = output_vec[N:]

        has_waypoint = wp_time >= 0

        # Plot Joint 1
        axes[idx, 0].plot(t, q1_traj, 'r-', linewidth=2, label='Trajectory')
        axes[idx, 0].plot(0, q1_start, 'go', markersize=10, label='Start', zorder=5)
        axes[idx, 0].plot(T, q1_end, 'ro', markersize=10, label='End', zorder=5)

        if has_waypoint:
            axes[idx, 0].plot(wp_time, wp_q1, 'b*', markersize=15, label='Waypoint', zorder=5)
            axes[idx, 0].axvline(wp_time, color='blue', linestyle='--', alpha=0.3)

        axes[idx, 0].set_ylabel('q1 (rad)', fontsize=11)
        title = f'Sample {sample_idx}: Joint 1'
        if has_waypoint:
            title += f' (waypoint at t={wp_time:.2f}s)'
        axes[idx, 0].set_title(title)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()

        # Plot Joint 2
        axes[idx, 1].plot(t, q2_traj, 'b-', linewidth=2, label='Trajectory')
        axes[idx, 1].plot(0, q2_start, 'go', markersize=10, label='Start', zorder=5)
        axes[idx, 1].plot(T, q2_end, 'ro', markersize=10, label='End', zorder=5)

        if has_waypoint:
            axes[idx, 1].plot(wp_time, wp_q2, 'b*', markersize=15, label='Waypoint', zorder=5)
            axes[idx, 1].axvline(wp_time, color='blue', linestyle='--', alpha=0.3)

        axes[idx, 1].set_ylabel('q2 (rad)', fontsize=11)
        title = f'Sample {sample_idx}: Joint 2'
        if has_waypoint:
            title += f' (waypoint at t={wp_time:.2f}s)'
        axes[idx, 1].set_title(title)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend()

        if idx == num_samples - 1:
            axes[idx, 0].set_xlabel('Time (s)', fontsize=11)
            axes[idx, 1].set_xlabel('Time (s)', fontsize=11)

    plt.suptitle('Optimized Trajectories with Waypoint Constraints', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSample visualization saved to: dataset_samples.png")


def split_dataset(dataset, train_ratio=0.8, save_splits=True):
    """Split dataset into training and testing sets"""
    inputs = dataset['inputs']
    outputs = dataset['outputs']

    num_samples = len(inputs)
    indices = np.random.permutation(num_samples)

    train_size = int(num_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = {
        'inputs': inputs[train_indices],
        'outputs': outputs[train_indices],
        'metadata': dataset['metadata']
    }

    test_data = {
        'inputs': inputs[test_indices],
        'outputs': outputs[test_indices],
        'metadata': dataset['metadata']
    }

    print(f"\nDataset split:")
    print(f"Training samples: {len(train_indices)} ({train_ratio * 100:.0f}%)")
    print(f"Testing samples: {len(test_indices)} ({(1 - train_ratio) * 100:.0f}%)")

    if save_splits:
        with open('train_dataset.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open('test_dataset.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        print("\nSplit datasets saved to: train_dataset.pkl and test_dataset.pkl")

    return train_data, test_data


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # ============================================
    # CONFIGURATION
    # ============================================
    NUM_SAMPLES = 15000  # Total number of trajectories
    BATCH_SIZE = 256  # Trajectories optimized in parallel (INCREASED)
    WAYPOINT_PROBABILITY = 0.5  # 50% of trajectories will have waypoints

    # Adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 4:
            BATCH_SIZE = 256
            print(f"Low GPU memory detected. Using batch size: {BATCH_SIZE}")
        elif gpu_memory_gb > 10:
            BATCH_SIZE = 4096
            print(f"High GPU memory detected. Using batch size: {BATCH_SIZE}")
        else:
            BATCH_SIZE = 1024
            print(f"Medium GPU memory. Using batch size: {BATCH_SIZE}")
    else:
        BATCH_SIZE = 32  # Smaller batch for CPU
        print("Running on CPU. Using smaller batch size.")

    print()

    # Generate dataset
    start_time = time.time()

    dataset = generate_dataset_pytorch(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        save_path='trajectory_dataset.pkl',
        waypoint_probability=WAYPOINT_PROBABILITY
    )

    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print(f"Average time per trajectory: {elapsed_time / NUM_SAMPLES:.3f} seconds")
    print(f"Trajectories per second: {NUM_SAMPLES / elapsed_time:.2f}")

    # Visualize samples
    visualize_samples(dataset, num_samples=4)

    # Split dataset
    train_data, test_data = split_dataset(dataset, train_ratio=0.8, save_splits=True)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • trajectory_dataset.pkl  (full dataset)")
    print("  • train_dataset.pkl       (80% training)")
    print("  • test_dataset.pkl        (20% testing)")
    print("  • dataset_samples.png     (visualization)")
    print("\nInput format: [q1_start, q1_end, q2_start, q2_end, waypoint_time, waypoint_q1, waypoint_q2]")
    print("Note: waypoint_time = -1 means no waypoint for that trajectory")
    print("\nReady for neural network training with PyTorch!")