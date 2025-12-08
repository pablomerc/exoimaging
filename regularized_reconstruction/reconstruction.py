'''
Regularized reconstruction of the image from the lightcurve.
Following equation 19 from Bouman et al. (2018).
'''

from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os


def generate_mu(image_size_pixels=10, fov_fraction=0.3):
    """
    Generates the mu image (circular Gaussian mean) described in the paper.

    Args:
        image_size_pixels (int): The width/height of the square image in pixels.
        fov_fraction (float): The standard deviation of the Gaussian as a fraction
                              of the total Field of View (width of the image).
                              The paper suggests 40-50% (0.4-0.5).

    Returns:
        numpy.ndarray: The 2D Gaussian image array.
    """

    # 1. Create a coordinate grid centered at (0,0)
    # We define the FOV as going from -0.5 to 0.5 (total width = 1.0 unit)
    x = np.linspace(-0.5, 0.5, image_size_pixels)
    y = np.linspace(-0.5, 0.5, image_size_pixels)
    X, Y = np.meshgrid(x, y)

    # 2. Define the standard deviation (sigma) based on the paper's specs
    # "Standard deviation of 40-50% of the reconstructed FOV"
    sigma = fov_fraction  # e.g., 0.45 corresponds to 45% of the total width

    # 3. Calculate the 2D Circular Gaussian function
    # formula: exp( - (x^2 + y^2) / (2 * sigma^2) )
    mu = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    return mu


def get_dft_matrix(image_size):
    """Returns the DFT operator matrix.
    Args:
    image_size: Height = width the image.
    dft_comps: A 1D array containing the indices of the rows of the full DFT
      matrix to keep.
    Returns:
    A 2D array representing the DFT measurement matrix, where only the rows
      corresponding to `dft_comps` are kept. The first half of the rows
      corresponds to the real part of the measurements, while the second
      half of the rows corresponds to the imaginary part.
    """
    dft_matrix_1d = scipy.linalg.dft(image_size)
    dft_matrix = np.kron(dft_matrix_1d, dft_matrix_1d)
    return dft_matrix
  # Split matrix into real and imaginary submatrices.
  # dft_matrix_expanded = np.concatenate(
  #     (dft_matrix.real, dft_matrix.imag), axis=0)
  # return dft_matrix_expanded


def genFreqComp(image_width, image_height):
    x_psize = 1 / image_width
    y_psize = 1 / image_height
    fN2 = int(np.floor(image_width/2)) #TODO: !!! THIS DOESNT WORK FOR ODD IMAGE SIZES
    fM2 = int(np.floor(image_height/2))

    ulist = (np.array([np.concatenate((np.linspace(0, fN2 - 1, fN2), np.linspace(-fN2, -1, fN2)), axis=0)])  / image_width ) / x_psize
    vlist = (np.array([np.concatenate((np.linspace(0, fM2 - 1, fM2), np.linspace(-fM2, -1, fM2)), axis=0)])  / image_height ) / y_psize

    ufull, vfull = np.meshgrid(ulist, vlist)

    # ufull = np.reshape(ufull, (im.xdim*im.ydim, -1), order='F')
    # vfull = np.reshape(vfull, (im.xdim*im.ydim, -1), order='F')
    ufull = np.reshape(ufull, (image_width*image_height, -1), order='F')
    vfull = np.reshape(vfull, (image_width*image_height, -1), order='F')

    return (ufull, vfull)


def genImCov(W, mu, a=2, frac=1/3, im_size=10):
    '''
    Generate the Lambda Matrix (image covariance)
    '''

    eps = 1e-3
    ufull, vfull = genFreqComp(im_size, im_size)
    uvdist = np.reshape( np.sqrt(ufull**2 + vfull**2), (ufull.shape[0]) ) + eps
    uvdist = uvdist / np.min(uvdist)
    # uvdist[0] = np.inf
    # instead, set it to eps
    # uvdist[0] = eps

    imCov_prime = np.dot( np.transpose(np.conj(W)) , np.dot( np.diag( 1/(uvdist**a) ), W ) )
    imCov = frac**2 * np.dot( np.diag(mu.reshape(-1)).T, np.dot(imCov_prime/imCov_prime[0,0], np.diag(mu.reshape(-1)) ) )

    # Make it be real and symmetric
    imCov = np.real(imCov)
    imCov = (imCov + imCov.T) / 2
    return imCov



def generate_R_matrix(num_measurements, sigma):
    """
    Generate the R matrix (Noise Covariance Matrix) for the inverse problem.

    As per Equation 9 in Bouman et al. (2018):
    R = diag[sigma[1]^2, ..., sigma[K]^2]

    If we assume isotropic noise (same noise level for all measurements),
    then R is a scaled identity matrix where the scalar is the VARIANCE (sigma^2).

    Args:
        num_measurements (int): The number of measurements (length of vector y).
        sigma (float): The standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: The R covariance matrix of size (num_measurements, num_measurements).
    """
    # The diagonal elements must be variance (sigma squared)
    variance = sigma ** 2

    # Create diagonal matrix
    R = np.eye(num_measurements) * variance

    return R


def visualize_result(x_hat, image, path_to_save, save=True, channel_names=['Red', 'Green', 'Blue'], show_combined=True):
    '''
    Plot side by side the target image and the reconstructed image.
    Handles both grayscale (2D) and RGB (3D) images.

    Args:
        x_hat: Reconstructed image, shape (H, W) for grayscale or (3, H, W) or (H, W, 3) for RGB
        image: Target image, shape (H, W) for grayscale or (3, H, W) or (H, W, 3) for RGB
        path_to_save: Path to save the comparison figure
        save: Whether to save the figure
        channel_names: Names for RGB channels (if applicable)
        show_combined: If True and RGB, also show combined color images
    '''
    # Handle different input shapes
    if x_hat.ndim == 3:
        if x_hat.shape[0] == 3:  # (3, H, W) format
            x_hat = np.transpose(x_hat, (1, 2, 0))  # Convert to (H, W, 3)
        # Otherwise assume (H, W, 3) format
        is_rgb = True
    else:
        is_rgb = False

    if image.ndim == 3:
        if image.shape[0] == 3:  # (3, H, W) format
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)
        # Otherwise assume (H, W, 3) format
    elif image.ndim == 2:
        # Convert grayscale to RGB for comparison
        image = np.stack([image] * 3, axis=-1)

    # Ensure values are in [0, 1] range
    print('x_hat range before clipping', x_hat.min(), x_hat.max())
    x_hat_clipped = np.clip(x_hat, 0, 1)
    x_hat_normalized = x_hat_clipped / x_hat_clipped.max()
    print('x_hat range after clipping', x_hat_normalized.min(), x_hat_normalized.max())
    image = np.clip(image, 0, 1)

    if is_rgb:
        if show_combined:
            # Show both combined RGB and individual channels
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            # Top row: Combined RGB images
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(x_hat_normalized)
            ax1.set_title('Reconstructed Image (RGB)', fontsize=12)
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(image)
            ax2.set_title('Target Image (RGB)', fontsize=12)
            ax2.axis('off')

            # Difference image
            ax3 = fig.add_subplot(gs[0, 2])
            diff = np.abs(x_hat - image)
            im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
            ax3.set_title('Absolute Difference', fontsize=12)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3)

            # Bottom row: Individual channels of reconstructed image
            for i, name in enumerate(channel_names):
                ax = fig.add_subplot(gs[1, i])
                im = ax.imshow(x_hat[:, :, i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Reconstructed - {name}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            # RGB visualization - channels only
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Reconstructed channels
            for i, (ax, name) in enumerate(zip(axes[0], channel_names)):
                im = ax.imshow(x_hat[:, :, i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Reconstructed - {name} Channel')
                ax.axis('off')
                plt.colorbar(im, ax=ax)

            # Target channels
            for i, (ax, name) in enumerate(zip(axes[1], channel_names)):
                im = ax.imshow(image[:, :, i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Target - {name} Channel')
                ax.axis('off')
                plt.colorbar(im, ax=ax)

        plt.tight_layout()
    else:
        # Grayscale visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(x_hat, cmap='gray', vmin=0, vmax=1)
        plt.title('Reconstructed Image')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.title('Target Image')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()

    if save:
        plt.savefig(path_to_save, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {path_to_save}")

    plt.show()

if __name__ == "__main__":

    path_to_lightcurve = 'outputs/lightcurve.csv'
    path_to_weights_matrix = 'outputs/weights_matrix.csv'
    # Try to load processed image, fallback to original if not found
    path_to_image = 'outputs/processed_image.png'
    if not os.path.exists(path_to_image):
        path_to_image = 'figures/earth.jpg'  # Fallback to original

    # Load data
    lightcurve = np.loadtxt(path_to_lightcurve, delimiter=',')
    weights_matrix = np.loadtxt(path_to_weights_matrix, delimiter=',')
    image = Image.open(path_to_image)

    # Handle lightcurve shape: could be (num_channels, num_timesteps) or (num_timesteps, num_channels)
    if lightcurve.shape[0] < lightcurve.shape[1]:
        # If first dimension is smaller, assume it's (num_channels, num_timesteps)
        lightcurve = lightcurve.T  # Transpose to (num_timesteps, num_channels)

    num_timesteps, num_channels = lightcurve.shape
    print(f"Loaded lightcurve: {num_timesteps} timesteps, {num_channels} channels")
    print(f"Loaded weights matrix: {weights_matrix.shape}")

    # Infer image size from weights_matrix
    # weights_matrix is (num_timesteps, num_pixels) when loaded from CSV
    num_pixels = weights_matrix.shape[1]
    image_size = int(np.sqrt(num_pixels))

    if image_size * image_size != num_pixels:
        raise ValueError(
            f"Cannot infer square image size from {num_pixels} pixels. "
            f"Expected a perfect square."
        )

    print(f"Inferred image size: {image_size}x{image_size}")

    # Generate priors (same for all channels) - use inferred size
    mu = generate_mu(image_size_pixels=image_size)
    W = get_dft_matrix(image_size)
    A = weights_matrix
    imCov = genImCov(W, mu, im_size=image_size)
    R = generate_R_matrix(num_timesteps, 0.0001)

    # Reconstruct each channel separately
    x_hat_channels = []

    # Suppress warnings from intermediate matrix operations
    # (these occur during BLAS computations but final results are valid)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Pre-compute matrices that are the same for all channels
        FLF = np.matmul(A, np.matmul(imCov, np.transpose(A)))
        RFLF = R + FLF
        invRFLF = np.linalg.pinv(RFLF)

        # Process each channel
        for channel_idx in range(num_channels):
            print(f"\nProcessing channel {channel_idx + 1}/{num_channels}...")
            y = lightcurve[:, channel_idx]  # Get measurements for this channel

            # Apply Equation 19: x_hat = mu + Lambda @ F.T @ inv(R + F @ Lambda @ F.T) @ (y - F @ mu)
            yFmu = (y - np.dot(A, mu.reshape(-1)))
            innovation = np.dot(invRFLF, yFmu)
            second_term = np.dot(imCov, np.dot(np.transpose(A), innovation))
            x_hat = mu.reshape(-1) + second_term

            # Reshape to image format - use inferred size
            x_hat_2d = np.reshape(np.real(x_hat), (image_size, image_size))
            x_hat_channels.append(x_hat_2d)

    # Stack channels: convert from list of (H, W) to (H, W, C)
    x_hat_rgb = np.stack(x_hat_channels, axis=-1)  # Shape: (image_size, image_size, 3)

    print(f"\nReconstructed image shape: {x_hat_rgb.shape}")

    # Convert target image to numpy array
    image_array = np.array(image)
    if image_array.ndim == 2:
        # If grayscale, convert to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[0] == 3:
        # If (3, H, W), transpose to (H, W, 3)
        image_array = np.transpose(image_array, (1, 2, 0))

    # Normalize target image to [0, 1]
    if image_array.max() > 1.0:
        image_array = image_array / 255.0

    # Visualize results
    visualize_result(x_hat_rgb, image_array, 'outputs/reconstructed_image_comparison.png')
