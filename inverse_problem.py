import torch
import numpy as np
from weights_matrix import simulate_circle_movement
import os
from config import (
    MATRIX_SIZE,
    NUM_TIMESTEPS,
    CIRCLE_RADIUS,
    EQUATOR_SHIFT,
    OUTPUT_DIR,
    LIGHTCURVE_PATH,
    WEIGHTS_MATRIX_PATH,
    RECOVERED_IMAGE_PATH,
    RECOVERED_IMAGE_PLOT_PATH,
)

# --- Paths ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

recompute_weights_matrix = False
load_weights_matrix = True

# --- 1. Load y from CSV ---
# y was saved as shape [3, 50] in your previous script
y_np = np.loadtxt(LIGHTCURVE_PATH, delimiter=",")   # → (3, 50)
y = torch.from_numpy(y_np).float()             # convert to torch tensor
print("Loaded y shape:", y.shape)              # expect: torch.Size([3, 50])

# Ensure y is 2D: (num_channels, num_timesteps)
if y.dim() == 1:
    y = y.unsqueeze(0)


if recompute_weights_matrix:
    # --- 2. Recompute A (weights_matrix) ---
    # This uses parameters from config.py to ensure consistency
    _, weights_matrix = simulate_circle_movement(
        matrix_size=MATRIX_SIZE,
        num_timesteps=NUM_TIMESTEPS,
        circle_radius=CIRCLE_RADIUS,
        equator_shift=EQUATOR_SHIFT
    )

    # weights_matrix is assumed to be (T, H, W) = (NUM_TIMESTEPS, MATRIX_SIZE, MATRIX_SIZE)
    T, H, W = weights_matrix.shape
    weights_matrix = torch.from_numpy(weights_matrix).float()
    print("Original A (weights_matrix) shape:", weights_matrix.shape)

if load_weights_matrix:
    #Alternatively, load weights_matrix from CSV
    weights_matrix = np.loadtxt(WEIGHTS_MATRIX_PATH, delimiter=",")
    weights_matrix = torch.from_numpy(weights_matrix).float()
    print("Loaded weights_matrix shape:", weights_matrix.shape)
    T = weights_matrix.shape[0]
    H, W = MATRIX_SIZE, MATRIX_SIZE


# Reshape weights_matrix from (T, H, W) to (T, H*W)
A = weights_matrix.view(T, -1)  # → (NUM_TIMESTEPS, MATRIX_SIZE*MATRIX_SIZE)
print("Reshaped A shape:", A.shape)

# --- 3. Compute pseudoinverse and solve y = A x ---
A_pinv = torch.linalg.pinv(A)
print("A_pinv shape:", A_pinv.shape)

x_est = y @ A_pinv.T                # (3, NUM_TIMESTEPS) @ (NUM_TIMESTEPS, MATRIX_SIZE*MATRIX_SIZE) → (3, MATRIX_SIZE*MATRIX_SIZE)
print("Recovered x_est (flattened) shape:", x_est.shape)

# --- 4. Reshape x back into images and save ---
x_est_images = x_est.view(y.shape[0], H, W)   # → (3, MATRIX_SIZE, MATRIX_SIZE)
print("Recovered x_est_images shape:", x_est_images.shape)

import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Save recovered image using torch utils
save_image(x_est_images, RECOVERED_IMAGE_PATH)

# Save recovered image as matplotlib figure
plt.figure(figsize=(5, 5))
plt.title("Recovered Image")
# Convert to H×W×C for plotting
plt.imshow(x_est_images.permute(1, 2, 0).clamp(0, 1))
plt.axis("off")
plt.savefig(RECOVERED_IMAGE_PLOT_PATH)
plt.show()

print(f"✅ Saved recovered image (torch) to {RECOVERED_IMAGE_PATH}")
print(f"✅ Saved recovered image (plot) to {RECOVERED_IMAGE_PLOT_PATH}")
