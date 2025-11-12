"""
Configuration file for exoimaging project.
Contains shared parameters that must be consistent across scripts.
"""

# Simulation parameters
MATRIX_SIZE = 10
NUM_TIMESTEPS = 100
CIRCLE_RADIUS = 5

# Image processing parameters
IMAGE_SIZE = (10, 10)  # (height, width) for image resizing

# Paths
OUTPUT_DIR = "outputs"
LIGHTCURVE_PATH = f"{OUTPUT_DIR}/lightcurve.csv"
WEIGHTS_MATRIX_PATH = f"{OUTPUT_DIR}/weights_matrix.csv"
PROCESSED_IMAGE_PATH = f"{OUTPUT_DIR}/processed_image.png"
PROCESSED_IMAGE_PLOT_PATH = f"{OUTPUT_DIR}/processed_image_plot.png"
LIGHTCURVE_PLOT_PATH = f"{OUTPUT_DIR}/lightcurve.png"
RECOVERED_IMAGE_PATH = f"{OUTPUT_DIR}/recovered_image.png"
RECOVERED_IMAGE_PLOT_PATH = f"{OUTPUT_DIR}/recovered_image_plot.png"
