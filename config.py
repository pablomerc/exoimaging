"""
Configuration file for exoimaging project.
Contains shared parameters that must be consistent across scripts.
"""

# Simulation parameters
MATRIX_SIZE = 34
NUM_TIMESTEPS = 500
CIRCLE_RADIUS = 15
EQUATOR_SHIFT = -5

NOISE_SIGMA = 0.01

# Image processing parameters
# IMAGE_SIZE should match MATRIX_SIZE for consistency
# If you want a different size, change MATRIX_SIZE instead
IMAGE_SIZE = (MATRIX_SIZE, MATRIX_SIZE)  # (height, width) for image resizing

# Paths
OUTPUT_DIR = "outputs"
LIGHTCURVE_PATH = f"{OUTPUT_DIR}/lightcurve.csv"
WEIGHTS_MATRIX_PATH = f"{OUTPUT_DIR}/weights_matrix.csv"
PROCESSED_IMAGE_PATH = f"{OUTPUT_DIR}/processed_image.png"
PROCESSED_IMAGE_PLOT_PATH = f"{OUTPUT_DIR}/processed_image_plot.png"
LIGHTCURVE_PLOT_PATH = f"{OUTPUT_DIR}/lightcurve.png"
RECOVERED_IMAGE_PATH = f"{OUTPUT_DIR}/recovered_image.png"
RECOVERED_IMAGE_PLOT_PATH = f"{OUTPUT_DIR}/recovered_image_plot.png"
