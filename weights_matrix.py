import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_circle_cell_overlap(circle_center_x, circle_center_y, circle_radius,
                                   cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                                   samples_per_cell=50):
    """
    Calculate the percentage of a cell that is covered by a circle.
    Uses numerical integration by sampling points within the cell.

    Returns: coverage percentage (0.0 to 1.0)
    """
    # Create a grid of sample points within the cell
    x_samples = np.linspace(cell_x_min, cell_x_max, samples_per_cell)
    y_samples = np.linspace(cell_y_min, cell_y_max, samples_per_cell)
    X, Y = np.meshgrid(x_samples, y_samples)

    # Calculate distance from each sample point to circle center
    distances = np.sqrt((X - circle_center_x)**2 + (Y - circle_center_y)**2)

    # Count points inside the circle
    points_inside = np.sum(distances <= circle_radius)
    total_points = samples_per_cell ** 2

    return points_inside / total_points


def simulate_circle_movement(matrix_size=10, num_timesteps=50, circle_radius=5,equator_shift=-2):
    """
    Simulate a circle moving from left to right along the equator of the matrix.

    Args:
        matrix_size: Size of the square matrix (default 10x10)
        num_timesteps: Number of timesteps to simulate
        circle_radius: Radius of the moving circle

    Returns:
        weights_matrix: Final weight matrix after all timesteps
        history: Tensor of shape (num_timesteps, matrix_size, matrix_size) with weight matrices at each timestep
    """
    # Initialize weights matrix (all start at 1.0 = fully uncovered)
    weights_matrix = np.ones((matrix_size, matrix_size))

    # Circle moves along the equator (middle row)
    equator_y = matrix_size / 2.0  # Center of middle row in continuous coordinates

    # Circle moves from left to right
    # Start slightly before the matrix, end slightly after
    x_start = -circle_radius
    x_end = matrix_size + circle_radius
    circle_x_positions = np.linspace(x_start, x_end, num_timesteps)

    history = []

    for timestep, circle_x in enumerate(circle_x_positions):
        weights_matrix = np.ones((matrix_size, matrix_size))
        # Calculate overlap for each cell in the matrix
        for i in range(matrix_size):
            for j in range(matrix_size):
                # Cell boundaries in continuous coordinates
                cell_x_min = j
                cell_x_max = j + 1
                cell_y_min = i
                cell_y_max = i + 1

                # Calculate overlap percentage
                coverage = calculate_circle_cell_overlap(
                    circle_x, equator_y + equator_shift, circle_radius,
                    cell_x_min, cell_x_max, cell_y_min, cell_y_max
                )

                # Subtract coverage from weight (clamp to 0 minimum)

                weights_matrix[i, j] = max(0.0, weights_matrix[i, j] - coverage)

        # Store history for visualization
        history.append(weights_matrix.copy())

        if (timestep + 1) % 10 == 0:
            print(f"Timestep {timestep + 1}/{num_timesteps} completed")

    # Convert history list to numpy tensor with shape (num_timesteps, matrix_size, matrix_size)
    history_tensor = np.array(history)

    return weights_matrix, history_tensor


def visualize_history(history, num_timesteps=50, matrix_size=10, save_animation=False):
    """
    Visualize the weight matrix evolution over time.

    Args:
        history: Tensor of shape (num_timesteps, matrix_size, matrix_size)
        num_timesteps: Number of timesteps
        matrix_size: Size of the matrix
        save_animation: If True, save an animated gif
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Grid of selected timesteps (top half)
    # Select key timesteps to display (every 5th timestep for 10 total)
    selected_timesteps = np.linspace(0, num_timesteps - 1, 10, dtype=int)
    cols = 5
    rows = 2

    # Create a grid for the timestep snapshots
    gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1.2, 1, 1])

    # Top section: timestep snapshots
    ax_top = fig.add_subplot(gs[0])
    ax_top.axis('off')
    ax_top.set_title('Weight Matrix Evolution - Selected Timesteps',
                     fontsize=14, fontweight='bold', pad=20)

    # Create subplots for each timestep
    inner_grid = fig.add_gridspec(rows, cols, left=0.05, right=0.95,
                                   top=0.95, bottom=0.7, hspace=0.3, wspace=0.3)

    for idx, t in enumerate(selected_timesteps):
        ax = fig.add_subplot(inner_grid[idx // cols, idx % cols])
        im = ax.imshow(history[t], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f't={t}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.set_ylabel('Row', fontsize=8)
        if idx >= 5:
            ax.set_xlabel('Col', fontsize=8)

    # Add colorbar for timestep snapshots
    cbar = fig.colorbar(im, ax=ax_top, orientation='horizontal',
                        pad=0.05, aspect=30, shrink=0.8)
    cbar.set_label('Weight (1=uncovered, 0=covered)', fontsize=9)

    # Plot 2: Mean weight over time
    ax2 = fig.add_subplot(gs[1])
    mean_weights = np.mean(history, axis=(1, 2))
    ax2.plot(mean_weights, linewidth=2, color='blue')
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Mean Weight', fontsize=11)
    ax2.set_title('Mean Weight Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])

    # Plot 3: Weight distribution at equator over time
    ax3 = fig.add_subplot(gs[2])
    equator_row = matrix_size // 2
    equator_weights = history[:, equator_row, :]  # Shape: (timesteps, width)

    im3 = ax3.imshow(equator_weights.T, cmap='RdYlGn', aspect='auto',
                     origin='lower', vmin=0, vmax=1, interpolation='bilinear')
    ax3.set_xlabel('Timestep', fontsize=11)
    ax3.set_ylabel('Column (along equator)', fontsize=11)
    ax3.set_title('Weight Evolution at Equator Row', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Weight', fontsize=9)

    plt.tight_layout()
    plt.savefig('weights_evolution.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'weights_evolution.png'")
    plt.show()

    # Optional: Create animation
    if save_animation:
        print("\nCreating animation...")
        fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

        # Create initial image and colorbar (only once)
        im_anim = ax_anim.imshow(history[0], cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax_anim.set_title(f'Timestep 0/{num_timesteps - 1}', fontsize=14, fontweight='bold')
        ax_anim.set_xlabel('Column', fontsize=11)
        ax_anim.set_ylabel('Row', fontsize=11)
        cbar_anim = plt.colorbar(im_anim, ax=ax_anim, label='Weight (1=uncovered, 0=covered)')

        def animate(frame):
            # Only update the image data, don't recreate everything
            im_anim.set_array(history[frame])
            ax_anim.set_title(f'Timestep {frame+1}/{num_timesteps}', fontsize=14, fontweight='bold')
            return [im_anim]

        anim = FuncAnimation(fig_anim, animate, frames=num_timesteps,
                            interval=100, blit=True, repeat=True)
        anim.save('weights_animation.gif', writer='pillow', fps=10)
        print("Animation saved as 'weights_animation.gif'")
        plt.close(fig_anim)


# Run simulation
if __name__ == "__main__":
    matrix_size = 10
    num_timesteps = 50
    circle_radius = 5
    equator_shift = -2

    print(f"Simulating circle movement:")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Circle radius: {circle_radius}")
    print(f"  Circle moves along equator (middle row)\n")

    final_weights, history = simulate_circle_movement(
        matrix_size=matrix_size,
        num_timesteps=num_timesteps,
        circle_radius=circle_radius,
        equator_shift=equator_shift
    )

    print(f"\nFinal weights matrix:")
    print(final_weights)
    print(f"\nMin weight: {final_weights.min():.4f}")
    print(f"Max weight: {final_weights.max():.4f}")
    print(f"Mean weight: {final_weights.mean():.4f}")

    print(f"\nHistory tensor shape: {history.shape}")
    print(f"History tensor dtype: {history.dtype}")
    print(f"History tensor range: [{history.min():.4f}, {history.max():.4f}]")

    # Visualize the history
    print("\nGenerating visualization...")
    visualize_history(history, num_timesteps=num_timesteps,
                     matrix_size=matrix_size, save_animation=True)
