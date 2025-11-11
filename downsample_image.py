import cv2 as cv
import sys
import os

path='figures/earth.jpg'
n_pixels = 10


## Load original image
img = cv.imread(path)
if img is None:
    print(f'Error: could not load image at {path}')
    sys.exit(1)
print('Original image shape',img.shape)


## Downsample the image
resized = cv.resize(img, (n_pixels,n_pixels), interpolation=cv.INTER_AREA)
print('Resized size:',resized.shape)


# --- Save the resized image ---
# You can control both the folder and filename:
save_path = f'figures/earth_{n_pixels}x{n_pixels}.jpg'
cv.imwrite(save_path, resized)

# Confirm that it worked
if os.path.exists(save_path):
    print(f"✅ Saved downsampled image to: {save_path}")
else:
    print("⚠️ Save failed!")

## Plot both
cv.namedWindow("Original", cv.WINDOW_NORMAL)
cv.namedWindow("Downsampled", cv.WINDOW_NORMAL)
cv.imshow("Original", img)
cv.imshow("Downsampled", cv.resize(resized, img.shape[1::-1], interpolation=cv.INTER_NEAREST))
cv.waitKey(0)
cv.destroyAllWindows()
