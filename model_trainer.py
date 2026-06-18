import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time
import os

# ---------------- GPU Setup ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("TRAINING SETUP")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)
print()


# ---------------- Dataset Class ----------------
class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data with normalization"""

    def __init__(self, data_path, normalize=True, stats=None):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.inputs = data['inputs'].astype(np.float32)
        self.outputs = data['outputs'].astype(np.float32)
        self.metadata = data['metadata']

        # Separate waypoint flag from continuous features
        self.has_waypoint = (self.inputs[:, 4] >= 0).astype(np.float32)

        # Normalize inputs
        if normalize:
            if stats is None:
                # Compute statistics from this dataset
                self.input_mean = np.mean(self.inputs[:, :4], axis=0)
                self.input_std = np.std(self.inputs[:, :4], axis=0) + 1e-8

                # For waypoint features (only normalize when waypoint exists)
                waypoint_data = self.inputs[self.has_waypoint == 1, 4:]
                if len(waypoint_data) > 0:
                    self.waypoint_mean = np.mean(waypoint_data, axis=0)
                    self.waypoint_std = np.std(waypoint_data, axis=0) + 1e-8
                else:
                    self.waypoint_mean = np.zeros(3)
                    self.waypoint_std = np.ones(3)

                self.output_mean = np.mean(self.outputs, axis=0)
                self.output_std = np.std(self.outputs, axis=0) + 1e-8
            else:
                # Use provided statistics (for test set)
                self.input_mean = stats['input_mean']
                self.input_std = stats['input_std']
                self.waypoint_mean = stats['waypoint_mean']
                self.waypoint_std = stats['waypoint_std']
                self.output_mean = stats['output_mean']
                self.output_std = stats['output_std']

            # Normalize
            self.inputs[:, :4] = (self.inputs[:, :4] - self.input_mean) / self.input_std

            # Normalize waypoint features only where waypoints exist
            mask = self.has_waypoint == 1
            if np.any(mask):
                self.inputs[mask, 4:] = (self.inputs[mask, 4:] - self.waypoint_mean) / self.waypoint_std

            self.outputs = (self.outputs - self.output_mean) / self.output_std
        else:
            self.input_mean = None
            self.input_std = None
            self.waypoint_mean = None
            self.waypoint_std = None
            self.output_mean = None
            self.output_std = None

        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)

        print(f"Loaded dataset from {data_path}")
        print(f"  Samples: {len(self.inputs)}")
        print(f"  Samples with waypoints: {int(self.has_waypoint.sum())}")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Output shape: {self.outputs.shape}")

    def get_stats(self):
        """Return normalization statistics"""
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'waypoint_mean': self.waypoint_mean,
            'waypoint_std': self.waypoint_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std
        }

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# ---------------- Neural Network Model ----------------
class TrajectoryMLP(nn.Module):
    """Multi-Layer Perceptron for trajectory prediction with improved architecture"""

    def __init__(self, input_dim=7, output_dim=100, hidden_dims=[256, 512, 1024, 512, 256]):
        super(TrajectoryMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers with residual connections
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)  # Increased dropout to combat overfitting
            ))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


# ---------------- Training Functions ----------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=100, save_path='best_model.pth', patience=15):
    """
    Train the model with early stopping

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of epochs
        save_path: Path to save best model
        patience: Early stopping patience
    """

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("Starting training...")
    print(f"Total epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print()

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"  → New best model saved! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses


# ---------------- Testing Functions ----------------
def test_model(model, test_loader, criterion, device, output_stats=None):
    """Test the model and compute detailed metrics"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_inputs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            total_loss += loss.item()

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    inputs = torch.cat(all_inputs, dim=0).numpy()

    # Denormalize if statistics provided
    if output_stats is not None:
        predictions = predictions * output_stats['output_std'] + output_stats['output_mean']
        targets = targets * output_stats['output_std'] + output_stats['output_mean']

    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # Per-joint metrics
    N = predictions.shape[1] // 2
    q1_pred = predictions[:, :N]
    q1_true = targets[:, :N]
    q2_pred = predictions[:, N:]
    q2_true = targets[:, N:]

    q1_rmse = np.sqrt(np.mean((q1_pred - q1_true) ** 2))
    q2_rmse = np.sqrt(np.mean((q2_pred - q2_true) ** 2))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss (MSE): {total_loss / len(test_loader):.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f} rad ({np.degrees(rmse):.2f}°)")
    print(f"Mean Absolute Error (MAE): {mae:.6f} rad ({np.degrees(mae):.2f}°)")
    print(f"Joint 1 RMSE: {q1_rmse:.6f} rad ({np.degrees(q1_rmse):.2f}°)")
    print(f"Joint 2 RMSE: {q2_rmse:.6f} rad ({np.degrees(q2_rmse):.2f}°)")
    print("=" * 60)

    return {
        'predictions': predictions,
        'targets': targets,
        'inputs': inputs,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'q1_rmse': q1_rmse,
        'q2_rmse': q2_rmse
    }


# ---------------- Visualization Functions ----------------
def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {save_path}")


def plot_predictions(test_results, metadata, num_samples=4, save_path='predictions.png', input_stats=None):
    """Plot predicted vs actual trajectories"""
    predictions = test_results['predictions']
    targets = test_results['targets']
    inputs = test_results['inputs']

    N = metadata['N']
    T = metadata['T']
    t = np.linspace(0, T, N)

    # Denormalize inputs to get original values
    if input_stats is not None:
        inputs_denorm = inputs.copy()
        # Denormalize first 4 values (joint angles)
        inputs_denorm[:, :4] = inputs[:, :4] * input_stats['input_std'] + input_stats['input_mean']
        # Denormalize waypoint data (only where waypoints exist)
        mask = inputs[:, 4] > -0.5  # waypoint exists (normalized value)
        if np.any(mask):
            inputs_denorm[mask, 4:] = (inputs[mask, 4:] * input_stats['waypoint_std'] +
                                       input_stats['waypoint_mean'])
        inputs = inputs_denorm

    # Select random samples
    indices = np.random.choice(len(predictions), size=min(num_samples, len(predictions)), replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(indices):
        input_vec = inputs[sample_idx]
        pred_vec = predictions[sample_idx]
        true_vec = targets[sample_idx]

        q1_start, q1_end, q2_start, q2_end = input_vec[:4]

        q1_pred = pred_vec[:N]
        q1_true = true_vec[:N]
        q2_pred = pred_vec[N:]
        q2_true = true_vec[N:]

        # Check for waypoint
        has_waypoint = len(input_vec) > 4 and input_vec[4] >= 0

        # Plot Joint 1
        axes[idx, 0].plot(t, q1_true, 'b-', linewidth=2.5, label='Optimized', alpha=0.7)
        axes[idx, 0].plot(t, q1_pred, 'r--', linewidth=2, label='Predicted')
        axes[idx, 0].plot(0, q1_start, 'go', markersize=10, label='Start', zorder=5)
        axes[idx, 0].plot(T, q1_end, 'ro', markersize=10, label='End', zorder=5)

        if has_waypoint:
            wp_time, wp_q1 = input_vec[4], input_vec[5]
            axes[idx, 0].plot(wp_time, wp_q1, 'b*', markersize=15, label='Waypoint', zorder=5)
            axes[idx, 0].axvline(wp_time, color='blue', linestyle='--', alpha=0.3)

        axes[idx, 0].set_ylabel('q1 (rad)', fontsize=11)
        axes[idx, 0].set_title(f'Sample {sample_idx}: Joint 1', fontsize=12)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend(fontsize=9)

        # Plot Joint 2
        axes[idx, 1].plot(t, q2_true, 'b-', linewidth=2.5, label='Optimized', alpha=0.7)
        axes[idx, 1].plot(t, q2_pred, 'r--', linewidth=2, label='Predicted')
        axes[idx, 1].plot(0, q2_start, 'go', markersize=10, label='Start', zorder=5)
        axes[idx, 1].plot(T, q2_end, 'ro', markersize=10, label='End', zorder=5)

        if has_waypoint:
            wp_time, wp_q2 = input_vec[4], input_vec[6]
            axes[idx, 1].plot(wp_time, wp_q2, 'b*', markersize=15, label='Waypoint', zorder=5)
            axes[idx, 1].axvline(wp_time, color='blue', linestyle='--', alpha=0.3)

        axes[idx, 1].set_ylabel('q2 (rad)', fontsize=11)
        axes[idx, 1].set_title(f'Sample {sample_idx}: Joint 2', fontsize=12)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend(fontsize=9)

        if idx == num_samples - 1:
            axes[idx, 0].set_xlabel('Time (s)', fontsize=11)
            axes[idx, 1].set_xlabel('Time (s)', fontsize=11)

    plt.suptitle('Predicted vs Optimized Trajectories', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Predictions plot saved to {save_path}")


def plot_error_distribution(test_results, save_path='error_distribution.png'):
    """Plot error distribution"""
    predictions = test_results['predictions']
    targets = test_results['targets']
    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall error distribution
    axes[0].hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prediction Error (rad)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Error Distribution', fontsize=13, fontweight='bold')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Error per timestep
    N = predictions.shape[1] // 2
    errors_per_timestep = np.mean(np.abs(errors), axis=0)

    axes[1].plot(errors_per_timestep[:N], 'r-', linewidth=2, label='Joint 1')
    axes[1].plot(errors_per_timestep[N:], 'b-', linewidth=2, label='Joint 2')
    axes[1].set_xlabel('Timestep', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error (rad)', fontsize=12)
    axes[1].set_title('Error vs Time', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Error distribution saved to {save_path}")


# ---------------- Main Training Pipeline ----------------
if __name__ == "__main__":
    # ============================================
    # CONFIGURATION
    # ============================================
    TRAIN_DATA_PATH = 'train_dataset.pkl'
    TEST_DATA_PATH = 'test_dataset.pkl'
    MODEL_SAVE_PATH = 'best_trajectory_model.pth'

    # Hyperparameters
    BATCH_SIZE = 32  # Reduced for better generalization
    NUM_EPOCHS = 200  # Increased epochs
    LEARNING_RATE = 0.0005  # Reduced learning rate
    WEIGHT_DECAY = 1e-4  # L2 regularization to prevent overfitting
    PATIENCE = 25  # Increased patience

    # Model architecture - Balanced network (not too big)
    HIDDEN_DIMS = [256, 512, 512, 256]  # Reduced from 1024

    # ============================================
    # Load Data
    # ============================================
    print("Loading datasets...")
    train_dataset = TrajectoryDataset(TRAIN_DATA_PATH, normalize=True)

    # Use same normalization stats for test set
    train_stats = train_dataset.get_stats()
    test_dataset = TrajectoryDataset(TEST_DATA_PATH, normalize=True, stats=train_stats)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print()
    print("Data normalization enabled:")
    print(f"  Input mean: {train_stats['input_mean']}")
    print(f"  Input std: {train_stats['input_std']}")
    print()

    # ============================================
    # Create Model
    # ============================================
    input_dim = train_dataset.inputs.shape[1]
    output_dim = train_dataset.outputs.shape[1]

    print(f"Creating model...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Hidden layers: {HIDDEN_DIMS}")

    model = TrajectoryMLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=HIDDEN_DIMS)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # ============================================
    # Setup Training
    # ============================================
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # AdamW with weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=5)

    # ============================================
    # Train Model
    # ============================================
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, save_path=MODEL_SAVE_PATH, patience=PATIENCE
    )

    # ============================================
    # Load Best Model and Test
    # ============================================
    print("\nLoading best model for testing...")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = test_model(model, test_loader, criterion, device, output_stats=train_stats)

    # ============================================
    # Visualizations
    # ============================================
    print("\nGenerating visualizations...")
    plot_training_history(train_losses, val_losses)
    plot_predictions(test_results, train_dataset.metadata, num_samples=4, input_stats=train_stats)
    plot_error_distribution(test_results)

    # ============================================
    # Save Test Results
    # ============================================
    results_dict = {
        'test_metrics': {
            'mse': test_results['mse'],
            'rmse': test_results['rmse'],
            'mae': test_results['mae'],
            'q1_rmse': test_results['q1_rmse'],
            'q2_rmse': test_results['q2_rmse']
        },
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    with open('test_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    print("\n" + "=" * 60)
    print("TRAINING AND TESTING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  • {MODEL_SAVE_PATH} (trained model)")
    print("  • test_results.pkl (test metrics)")
    print("  • training_history.png")
    print("  • predictions.png")
    print("  • error_distribution.png")
    print("=" * 60)