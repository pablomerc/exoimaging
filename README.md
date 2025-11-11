# Exoplanet Imaging Project

This project simulates and solves the inverse problem of reconstructing an exoplanet's surface image from transit lightcurve observations. It demonstrates both the forward problem (generating lightcurves from images) and the inverse problem (reconstructing images from lightcurves).

## Overview

The project models how a planet transiting across a star creates a time-varying lightcurve. By observing how the brightness changes over time, we can attempt to reconstruct the planet's surface features using linear algebra techniques.

## Project Structure

```
exoimaging/
├── weights_matrix.py          # Simulates circle movement (transit) and generates weight matrices
├── run_forward.py             # Forward problem: generates lightcurve from an image
├── inverse_problem.py         # Inverse problem: reconstructs image from lightcurve
├── downsample_image.py        # Utility script to downsample images
├── figures/                   # Input images and generated visualizations
└── outputs/                   # Generated data files (lightcurves, weights, reconstructed images)
```

## How It Works

### 1. Weight Matrix Generation (`weights_matrix.py`)

Simulates a circular object (representing a planet) moving across a grid (representing the star's surface). For each timestep, it calculates how much of each grid cell is covered by the circle, creating a weight matrix.

- **Key Function**: `simulate_circle_movement()`
  - Simulates a circle moving from left to right along the equator
  - Calculates overlap between the circle and each grid cell using numerical integration
  - Returns a history tensor of shape `(num_timesteps, matrix_size, matrix_size)`

- **Visualization**: Includes functions to visualize the weight matrix evolution over time and create animated GIFs

### 2. Forward Problem (`run_forward.py`)

Generates a lightcurve from an input image:

1. **Load and Process Image**:
   - Loads an image (e.g., `figures/earth.jpg`)
   - Resizes it to 10×10 pixels (3 color channels: RGB)

2. **Apply Transit Simulation**:
   - Uses the weight matrix to simulate how the planet transits across the star
   - Computes: `lightcurve = image @ weights_matrix.T`
   - Each timestep represents how much light is blocked at that moment

3. **Output**:
   - Saves lightcurve to `outputs/lightcurve.csv` (shape: 3×50 for RGB channels and 50 timesteps)
   - Saves weights matrix to `outputs/weights_matrix.csv`
   - Saves processed image to `outputs/processed_image.png`

### 3. Inverse Problem (`inverse_problem.py`)

Reconstructs the original image from the lightcurve:

1. **Load Data**:
   - Loads the lightcurve (`y`) from CSV
   - Loads the weights matrix (`A`) from CSV

2. **Solve Linear System**:
   - Solves `y = A x` where:
     - `y`: lightcurve (3×50) - observed brightness over time
     - `A`: weights matrix (50×100) - flattened weight matrices
     - `x`: reconstructed image (3×100) - flattened image pixels
   - Uses pseudoinverse: `x = y @ A_pinv.T`

3. **Visualize Result**:
   - Reshapes the solution back to image format (3×10×10)
   - Displays the reconstructed image

### 4. Image Downsampling (`downsample_image.py`)

Utility script to downsample images to a specific resolution (default: 10×10 pixels) using OpenCV.

## Usage

### Running the Forward Problem

```bash
python run_forward.py
```

This will:
- Process `figures/earth.jpg` (resize to 10×10)
- Generate a lightcurve using the transit simulation
- Save outputs to the `outputs/` directory

### Running the Inverse Problem

```bash
python inverse_problem.py
```

This will:
- Load the lightcurve and weights matrix from `outputs/`
- Solve the inverse problem to reconstruct the image
- Display the reconstructed image

### Generating Weight Matrices

```bash
python weights_matrix.py
```

This will:
- Simulate the circle movement
- Generate visualizations of the weight evolution
- Optionally create an animated GIF

### Downsampling Images

```bash
python downsample_image.py
```

This will downsample `figures/earth.jpg` to 10×10 pixels and save it.

## Dependencies

- `torch` / `pytorch` - Tensor operations and linear algebra
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `torchvision` / `PIL` - Image processing
- `opencv-python` (cv2) - Image downsampling (for `downsample_image.py`)

## Key Parameters

- **Matrix Size**: 10×10 grid (configurable)
- **Number of Timesteps**: 50 (configurable)
- **Circle Radius**: 1.5-4.0 (configurable, affects transit coverage)
- **Image Channels**: 3 (RGB)

## Output Files

- `outputs/lightcurve.csv`: Time-series brightness data (3×50)
- `outputs/weights_matrix.csv`: Weight matrices for each timestep (50×100)
- `outputs/processed_image.png`: The downsampled input image
- `figures/weights_evolution.png`: Visualization of weight matrix evolution
- `figures/weights_animation.gif`: Animated visualization of the transit

## Notes

- The current implementation uses a simple pseudoinverse approach, which works for well-conditioned systems
- The transit is simulated as a circle moving horizontally across the middle row (equator)
- The reconstruction quality depends on the conditioning of the weight matrix and the number of observations
