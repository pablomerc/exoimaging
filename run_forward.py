import torchvision.io as io
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the plot
plt.savefig(LIGHTCURVE_PLOT_PATH)
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
print(f"✅ Saved lightcurve to {LIGHTCURVE_PATH}")
print(f"✅ Saved weights matrix to {WEIGHTS_MATRIX_PATH}")
print(f"✅ Saved processed image (torch) to {PROCESSED_IMAGE_PATH}")
print(f"✅ Saved processed image (plot) to {PROCESSED_IMAGE_PLOT_PATH}")
