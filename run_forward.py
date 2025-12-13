import torchvision.io as io
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from torchvision.utils import save_image
from config import (
    MATRIX_SIZE,
    NUM_TIMESTEPS,
    CIRCLE_RADIUS,
    EQUATOR_SHIFT,
    IMAGE_SIZE,
    OUTPUT_DIR,
    LIGHTCURVE_PATH,
    WEIGHTS_MATRIX_PATH,
    PROCESSED_IMAGE_PATH,
    PROCESSED_IMAGE_PLOT_PATH,
    LIGHTCURVE_PLOT_PATH,
    CIRCLE_GEOMETRY_PLOT_PATH,
    NOISE_SIGMA
)

image_path = 'figures/earth.jpg'

image_tensor = io.read_image(image_path)

# If you need to convert it to a float tensor and normalize (e.g., for model input)
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert tensor to PIL Image first for some transforms
    transforms.Resize(IMAGE_SIZE), # Resize to configured image size
    transforms.ToTensor(), # Converts to float tensor (0.0-1.0)
])
processed_image = transform(image_tensor)

print(f"Original image tensor shape: {image_tensor.shape}")
print(f"Processed image tensor shape: {processed_image.shape}")


### Forward pass

from weights_matrix import simulate_circle_movement

_,weights_matrix = simulate_circle_movement(
    matrix_size=MATRIX_SIZE,
    num_timesteps=NUM_TIMESTEPS,
    circle_radius=CIRCLE_RADIUS,
    equator_shift=EQUATOR_SHIFT
)
weights_matrix = torch.tensor(weights_matrix, dtype=processed_image.dtype, device=processed_image.device)



print(weights_matrix.shape)
print(processed_image.shape)

T = weights_matrix.shape[0]
C = processed_image.shape[0]
H_weights, W_weights = weights_matrix.shape[1], weights_matrix.shape[2]
H_image, W_image = processed_image.shape[1], processed_image.shape[2]

# Check for size mismatch
if H_weights != H_image or W_weights != W_image:
    raise ValueError(
        f"Size mismatch: weights_matrix is {H_weights}x{W_weights} but "
        f"processed_image is {H_image}x{W_image}. "
        f"Make sure IMAGE_SIZE in config.py matches MATRIX_SIZE."
    )

weights_matrix = torch.tensor(weights_matrix).view(T, -1)
processed_image_flattened = processed_image.view(C, -1)

print(f"Weights matrix shape (flattened): {weights_matrix.shape}")
print(f"Processed image shape (flattened): {processed_image_flattened.shape}")

result = processed_image_flattened @ weights_matrix.T

amplitude=result.max() - result.min()
noise=NOISE_SIGMA * amplitude
result=result + torch.randn_like(result) * noise

# Create figure with extra space on the right for text box
fig = plt.figure(figsize=(10, 6))
ax = plt.gca()

plt.plot(result[0])
plt.plot(result[1])
plt.plot(result[2])
plt.xlabel("Timestep")
plt.ylabel("Brightness")
if NOISE_SIGMA > 0:
    plt.title(f"Lightcurve with Noise (sigma={NOISE_SIGMA})")
else:
    plt.title("Lightcurve")
plt.legend(["Red", "Green", "Blue"])

# Adjust subplot to leave space on the right for text box
plt.subplots_adjust(right=0.75)

# Add text box with simulation parameters outside the plot area
textstr = f'# points: {NUM_TIMESTEPS}\nr occluder: {CIRCLE_RADIUS}\noffset: {EQUATOR_SHIFT}\nimage size: {MATRIX_SIZE}x{MATRIX_SIZE}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.77, 0.5, textstr, fontsize=10, verticalalignment='center',
         bbox=props, transform=fig.transFigure)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the plot
plt.savefig(LIGHTCURVE_PLOT_PATH)
plt.show()

# Create visualization of the circle geometry
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-MATRIX_SIZE/2 - CIRCLE_RADIUS, MATRIX_SIZE/2 + CIRCLE_RADIUS)
ax.set_ylim(-MATRIX_SIZE/2 - CIRCLE_RADIUS, MATRIX_SIZE/2 + CIRCLE_RADIUS)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

# Draw the occluded circle (image boundary) - radius = MATRIX_SIZE/2
occluded_circle = patches.Circle((0, 0), MATRIX_SIZE/2, fill=False,
                                 edgecolor='blue', linewidth=3, linestyle='-')
ax.add_patch(occluded_circle)
# Add label for occluded circle
ax.text(0, MATRIX_SIZE/2 + 2, 'Occluded (Image Size)', ha='center',
        fontsize=11, fontweight='bold', color='blue',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Draw the occultor circle (moving circle) - radius = CIRCLE_RADIUS
# Position it at a representative location along the equator
equator_y = EQUATOR_SHIFT
occultor_x = 0  # Center position for visualization
occultor_circle = patches.Circle((occultor_x, equator_y), CIRCLE_RADIUS,
                                 fill=False, edgecolor='red', linewidth=3,
                                 linestyle='--')
ax.add_patch(occultor_circle)
# Add label for occultor circle
ax.text(occultor_x, equator_y + CIRCLE_RADIUS + 2, 'Occultor', ha='center',
        fontsize=11, fontweight='bold', color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Mark the center
ax.plot(0, 0, 'ko', markersize=8, zorder=5)

ax.set_xlabel('X position', fontsize=11)
ax.set_ylabel('Y position', fontsize=11)
ax.set_title(f'Circle Geometry\nImage size: {MATRIX_SIZE}x{MATRIX_SIZE}, '
             f'Occultor radius: {CIRCLE_RADIUS}, Equator shift: {EQUATOR_SHIFT}',
             fontsize=12, fontweight='bold')
ax.set_aspect('equal')

# Save the circle geometry plot
plt.savefig(CIRCLE_GEOMETRY_PLOT_PATH, dpi=150, bbox_inches='tight')
plt.show()

# Save the tensor as a CSV
np.savetxt(LIGHTCURVE_PATH, result.cpu().numpy(), delimiter=",")
np.savetxt(WEIGHTS_MATRIX_PATH, weights_matrix.cpu().numpy(), delimiter=",")

# Save processed image using torch utils
save_image(processed_image, PROCESSED_IMAGE_PATH)

# Save processed image as matplotlib figure
plt.figure(figsize=(5, 5))
plt.title("Processed Image")
plt.imshow(processed_image.permute(1, 2, 0).clamp(0, 1))
plt.axis("off")
plt.savefig(PROCESSED_IMAGE_PLOT_PATH)
plt.close()  # Close the figure to avoid showing it

print(f"✅ Saved lightcurve plot to {LIGHTCURVE_PLOT_PATH}")
print(f"✅ Saved circle geometry plot to {CIRCLE_GEOMETRY_PLOT_PATH}")
print(f"✅ Saved lightcurve to {LIGHTCURVE_PATH}")
print(f"✅ Saved weights matrix to {WEIGHTS_MATRIX_PATH}")
print(f"✅ Saved processed image (torch) to {PROCESSED_IMAGE_PATH}")
print(f"✅ Saved processed image (plot) to {PROCESSED_IMAGE_PLOT_PATH}")
