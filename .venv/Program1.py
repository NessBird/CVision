# CS 510 Comp Vision
# Ness Blackbird
# Program 1.1
import cv2
import numpy as np

# Part 1i. Program to apply a Gaussian filter.
def convolve(image, kernel):
    # It's grayscale, so there should be just two dimensions, but there are three?
    height, width = image.shape

    # Size of the square kernel.
    kernel_size = kernel.shape[0]
    # Add a zero border, size 1 or 2, so the convolution operation doesn't run into an error.
    border = 1 if kernel_size == 3 else 2
    out = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

    # Make a copy of the image, blurred using the filter.
    for y in range(border, height - border * 2):
        for x in range(border, width - border * 2):
            # Calculate this pixel: Calculate the Hadamard product of the kernel and this
            # segment of the input image, then sum the resulting submatrix.
            out[y, x] = np.sum(image[y:y + kernel_size, x:x + kernel_size] * kernel)
    return out

img1 = cv2.imread('/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/filter1_img.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/filter2_img.jpg', cv2.IMREAD_GRAYSCALE)

# Convert everything to float to allow for values > 255. We'll convert it back later, before we display.
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)

# Original images first. Put everything in a list for later display.
out_images = []
out_images.append([img1, "Image 1 original"])
out_images.append([img2, "Image 2 original"])

# Now Gaussian filter 1.
filter = 1.0/16.0 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float32')
out_images.append([convolve(img1, filter), "Image 1 filter 1"])
out_images.append([convolve(img2, filter), "Image 2 filter 1"])

# Gaussian filter 2.
filter = 1/273 * np.array([[1, 4, 7, 4, 1],
                  [4, 16, 26, 16, 4],
                  [7, 26, 41, 26, 7],
                  [4, 16, 26, 16, 4],
                  [1, 4, 7, 4, 1]], dtype='float32')
out_images.append([convolve(img1, filter), "Image 1 filter 2"])
out_images.append([convolve(img2, filter), "Image 2 filter 2"])

# "Derivative of Gaussian" filters. Img 1:
xf = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
yf = np.array([[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]], dtype='float32')

dx = convolve(img1, xf)
dy = convolve(img1, yf)
# Calculate the magnitude of the two filters combined.
d = np.sqrt((dx * dx) + (dy * dy))
# Normalize it (the magnitudes can get over 255) and turn it into an image.
d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
out_images.append([d, "Image 1 DoG"])

# This one doesn't get displayed. It's directions, not pixels.
sobel1 = np.arctan2(dy, dx)

# Do it all again for Img 2:
dx = convolve(img2, xf)
dy = convolve(img2, yf)
d = np.sqrt((dx * dx) + (dy * dy))
d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
out_images.append([dy, "Image 2 DoG"])

sobel2 = np.arctan2(dy, dx)


n = 1
for image in out_images:
    # Convert back to uint8. When CV2 gets float values, it assumes they're 0-1, and if they're not,
    # its operations are undefined. Jeez. Throw an error? Please?
    image[0] = image[0].astype(np.uint8)
    cv2.imshow(image[1], image[0])
    n += 1
# Force imshow to actually display the image.
cv2.waitKey(10)
input("Press any key to finish")
cv2.destroyAllWindows()