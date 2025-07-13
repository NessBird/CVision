# CS 510 Comp Vision
# Ness Blackbird
# Program 1.1
import cv2
import numpy as np

# Part 1i. Program to apply a Gaussian filter.
def convolve(img, kernel):
    # It's grayscale, so there should be just two dimensions, but there are three?
    height, width = img.shape

    # Size of the square kernel.
    kernel_size = kernel.shape[0]
    # Add a zero border, size 1 or 2, so the convolution operation doesn't run into an error.
    border = 1 if kernel_size == 3 else 2
    padded = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
    out = np.zeros_like(img)

    # Make a copy of the image, blurred using the filter.
    for y1 in range(border, padded.shape[0] - border - 1):
        for x1 in range(border, padded.shape[1] - border - 1):
            # These are the positions we're starting with -- the top left of the convolution window.
            y2 = y1 - border
            x2 = x1 - border
            # Calculate this pixel: Calculate the Hadamard product of the kernel and this
            # segment of the input image, then sum the resulting submatrix.
            # Here, in terms of out[], we're starting with 0, 0 and going from there.
            out[y2, x2] = np.sum(padded[y2:y2 + kernel_size, x2:x2 + kernel_size] * kernel)
    return out

d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/'
img1 = cv2.imread(d + 'filter1_img.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(d + 'filter2_img.jpg', cv2.IMREAD_GRAYSCALE)

# Convert everything to float to allow for values > 255. We'll convert it back later, before we display.
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)

# Put everything in a list for later display.
out_images = []

# Now Gaussian filter 1.
flt = 1.0 / 16.0 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float32')
out_images.append([convolve(img1, flt), "filter1_g1"])
out_images.append([convolve(img2, flt), "filter2_g1"])

# Gaussian filter 2.
flt = 1 / 273 * np.array([[1, 4, 7, 4, 1],
                          [4, 16, 26, 16, 4],
                          [7, 26, 41, 26, 7],
                          [4, 16, 26, 16, 4],
                          [1, 4, 7, 4, 1]], dtype='float32')
out_images.append([convolve(img1, flt), "filter1_g2"])
out_images.append([convolve(img2, flt), "filter2_g2"])

# "Derivative of Gaussian" filters. Img 1:
xf = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
yf = np.array([[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]], dtype='float32')

dx = convolve(img1, xf)
dy = convolve(img1, yf)
# Calculate the magnitude of the two filters combined.
dt = np.sqrt((dx * dx) + (dy * dy))
# Normalize them (the magnitudes can get over 255) and turn them into images.
dt = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX)
out_images.append([dx, "filter1_gx"])
out_images.append([dy, "filter1_gy"])
out_images.append([dt, "filter1_sobel"])

# Do it all again for Img 2:
dx = convolve(img2, xf)
dy = convolve(img2, yf)
dt = np.sqrt((dx * dx) + (dy * dy))
dt = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX)
out_images.append([dx, "filter2_gx"])
out_images.append([dy, "filter2_gy"])
out_images.append([dt, "filter2_sobel"])


n = 1
for img in out_images:
    # Convert back to uint8. When CV2 gets float values, it assumes they're 0-1, and if they're not,
    # its operations are undefined. Jeez. Throw an error? Please?
    img[0] = img[0].astype(np.uint8)
    cv2.imshow(img[1], img[0])
    cv2.imwrite(d + img[1] + '.jpg', img[0])
    n += 1
    # Force imshow to actually display the image.
    cv2.waitKey(100)
input("Press any key to finish")
cv2.destroyAllWindows()