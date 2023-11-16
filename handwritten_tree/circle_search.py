import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage import io, color
from scipy import ndimage as ndi
from sklearn.cluster import MeanShift

# Load the image
input_image = io.imread('rvosuke.png', as_gray=True)
# 转化为3通道图像


def detect_circles(image):
    """
    This function detects circles in an image using Hough Transform.
    :param image: Input image in which circles need to be detected
    :return: Image with detected circles
    """
    # Convert rgba to grayscale
    # gray_image = color.rgb2gray(image)

    # Edge detection
    edges = canny(image, sigma=3)

    # Detect circles
    hough_radii = np.arange(15, 30, 2)  # range of radii to search for
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=10)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image_with_circles = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image_with_circles[circy, circx] = (220, 20, 20)

    # Display the result
    ax.imshow(image_with_circles, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    return image_with_circles


# Run the circle detection algorithm on the uploaded image
detected_circles = detect_circles(input_image)
