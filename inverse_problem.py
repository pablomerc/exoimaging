import torch
import numpy as np
from weights_matrix import simulate_circle_movement
import os

# --- Paths ---
LIGHTCURVE_PATH = "outputs/lightcurve.csv"
os.makedirs("outputs", exist_ok=True)

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
    # This must match the parameters you used before
    _, weights_matrix = simulate_circle_movement(
        matrix_size=10,
        num_timesteps=50,
        circle_radius=1.5
    )

    # weights_matrix is assumed to be (T, H, W) = (50, 10, 10)
    T, H, W = weights_matrix.shape
    weights_matrix = torch.from_numpy(weights_matrix).float()
    print("Original A (weights_matrix) shape:", weights_matrix.shape)

if load_weights_matrix:
    #Alternatively, load weights_matrix from CSV
    weights_matrix = np.loadtxt("outputs/weights_matrix.csv", delimiter=",")
    weights_matrix = torch.from_numpy(weights_matrix).float()
    print("Loaded weights_matrix shape:", weights_matrix.shape)
    T = weights_matrix.shape[0]
    H,W=10,10


# 50, 10, 10
A = weights_matrix.view(T, -1)  # → (50, 100)
print("Reshaped A shape:", A.shape)

# --- 3. Compute pseudoinverse and solve y = A x ---
# A: (50, 100), y: (3, 50)
# Pseudoinverse A⁺ has shape (100, 50), and x = y @ (A⁺)ᵀ → (3, 100)
A_pinv = torch.linalg.pinv(A)       # → (100, 50)
print("A_pinv shape:", A_pinv.shape)

x_est = y @ A_pinv.T                # (3, 50) @ (50, 100) → (3, 100)
print("Recovered x_est (flattened) shape:", x_est.shape)

# --- 4. Reshape x back into images and save ---
x_est_images = x_est.view(y.shape[0], H, W)   # → (3, 10, 10)
print("Recovered x_est_images shape:", x_est_images.shape)

import matplotlib.pyplot as plt
import torchvision

# x_est_images is your (3,10,10) tensor
plt.figure(figsize=(5, 5))
plt.title("Recovered Image")

# Convert to H×W×C for plotting
plt.imshow(x_est_images.permute(1, 2, 0).clamp(0, 1))
plt.axis("off")
plt.show()

# Save as torch tensor and numpy
# torch.save(x_est_images, "outputs/x_est_images.pt")
# np.save("outputs/x_est_images.npy", x_est_images.numpy())
# print("Saved x_est_images to outputs/x_est_images.pt and outputs/x_est_images.npy")
