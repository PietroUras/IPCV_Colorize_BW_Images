import numpy as np
import cv2

# This script performs a series of image preprocessing operations to evaluate how different techniques
# affect the input image before colorization. It includes three main operations: histogram equalization,
# denoising, and removal of grain and scratches. These operations can help "fix" or improve the quality
# of an old or degraded image.

def equalize_bgr_image(image):
    """
    Applies histogram equalization to each channel of a BGR image.

    Args:
        image (numpy.ndarray): The input BGR image.

    Returns:
        numpy.ndarray: The BGR image with equalized histograms for each channel.
    """
    # Split the image into B, G, R channels
    B, G, R = cv2.split(image)

    # Apply histogram equalization to each channel
    equalized_B = cv2.equalizeHist(B)
    equalized_G = cv2.equalizeHist(G)
    equalized_R = cv2.equalizeHist(R)

    # Merge the channels back into a BGR image
    equalized_image = cv2.merge((equalized_B, equalized_G, equalized_R))
    return equalized_image


def simple_denoise(image, kernel_size=15, sigma=0):
    """
    Applies low-pass filtering (Gaussian Blur) to reduce grain in an old photo.

    Args:
        image (np.ndarray): The input BGR image.
        kernel_size (int): Size of the Gaussian kernel (must be odd). Default is 15 to remove grain from high resolution input.
        sigma (float): Standard deviation for Gaussian kernel.
                       If 0, it is calculated automatically. Default is 0.

    Returns:
        np.ndarray: The denoised image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a BGR image (3 channels).")

    # Apply Gaussian Blur to the image
    denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return denoised_image


def remove_grain_and_scratches(image, kernel_size=7, sigma=0, morph_kernel_size=5):
    """
    Reduces grain and removes scratches from an old photo using Gaussian Blur,
    median filtering, and morphological operations.

    Args:
        image (np.ndarray): The input BGR image.
        kernel_size (int): Size of the Gaussian kernel (must be odd). Default is 7.
        sigma (float): Standard deviation for Gaussian kernel. If 0, it is calculated automatically.
        morph_kernel_size (int): Size of the morphological kernel for removing scratches. Default is 5.

    Returns:
        np.ndarray: The cleaned image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a BGR image (3 channels).")

    # Step 1: Reduce grain using Gaussian Blur
    denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Step 2: Further reduce grain using Median Blur
    denoised_image = cv2.medianBlur(denoised_image, kernel_size)

    # Step 3: Convert image to grayscale for scratch removal
    gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Detect scratches using morphological operations
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))

    # Closing (dilate, then erode) to fill in scratches
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, morph_kernel)

    # Opening (erode, then dilate) to remove small white noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, morph_kernel)

    # Step 5: Combine cleaned grayscale with original color
    restored_image = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

    return restored_image
