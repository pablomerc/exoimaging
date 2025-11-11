import torchvision.io as io
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

image_path = 'figures/earth.jpg'

image_tensor = io.read_image(image_path)

# If you need to convert it to a float tensor and normalize (e.g., for model input)
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert tensor to PIL Image first for some transforms
    transforms.Resize((10, 10)), # Example: Resize to 224x224
    transforms.ToTensor(), # Converts to float tensor (0.0-1.0)
])
processed_image = transform(image_tensor)

print(f"Original image tensor shape: {image_tensor.shape}")
print(f"Processed image tensor shape: {processed_image.shape}")

# import matplotlib.pyplot as plt
# import torchvision

# # --- Display the original image ---
# plt.figure(figsize=(5, 5))
# plt.title("Original Image")
# plt.imshow(image_tensor.permute(1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
# plt.axis("off")
# plt.show()


# plt.figure(figsize=(5, 5))
# plt.title("Processed Image (Resized)")
# plt.imshow(torchvision.transforms.functional.to_pil_image(processed_image))
# plt.axis("off")
# plt.show()


### Forward pass

from weights_matrix import simulate_circle_movement

_,weights_matrix = simulate_circle_movement(matrix_size=10, num_timesteps=50, circle_radius=4)
weights_matrix = torch.tensor(weights_matrix, dtype=processed_image.dtype, device=processed_image.device)



print(weights_matrix.shape)
print(processed_image.shape)

T=weights_matrix.shape[0]
C=processed_image.shape[0]

weights_matrix = torch.tensor(weights_matrix).view(T,-1)
processed_image_flattened = processed_image.view(C,-1)

print(weights_matrix.shape)
print(processed_image_flattened.shape)

result = processed_image_flattened @ weights_matrix.T

plt.plot(result[0])
plt.plot(result[1])
plt.plot(result[2])
plt.xlabel("Timestep")
plt.ylabel("Brightness")
plt.title("Lightcurve")
plt.legend(["Red", "Green", "Blue"])

# Create the output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save the plot
plt.savefig("outputs/lightcurve.png")
plt.show()

# Save the tensor as a CSV
np.savetxt("outputs/lightcurve.csv", result.cpu().numpy(), delimiter=",")
np.savetxt("outputs/weights_matrix.csv", weights_matrix.cpu().numpy(), delimiter=",")
#save processed image in outputs
save_image(processed_image, "outputs/processed_image.png")
print("✅ Saved lightcurve plot to outputs/lightcurve.png")
print("✅ Saved lightcurve to outputs/lightcurve.csv")
print("✅ Saved weights matrix to outputs/weights_matrix.csv")
print("✅ Saved processed image to outputs/processed_image.png")
