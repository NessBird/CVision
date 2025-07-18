# Program 2: Optical Flow.
# Ness Blackbird for Computer Vision 510.

import cv2
import numpy as np

def convolve(img, kernel):
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

def optical_flow(imgA, imgB, outname):
    # Calculate optical flow matrices between the two images.

    # Sobel filters.
    xf = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
    yf = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')

    height, width = imgA.shape

    # Get the x and y gradients.
    gx = convolve(imgA, xf)
    gy = convolve(imgA, yf)

    # Add borders, they're going to need them.
    gx = cv2.copyMakeBorder(gx, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    gy = cv2.copyMakeBorder(gy, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Minimum determinant to bother with.
    epsilon = 0.0001

    # Now pad the borders of the images, first and second frames.
    first = cv2.copyMakeBorder(imgA, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    second = cv2.copyMakeBorder(imgB, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Calculate the difference between the two images.
    dt = second - first

    # Now we do the Lucas-Kanade calculation for each pixel, excluding the borders.
    # Set up an output image that will be in color.
    out = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(1, first.shape[0] - 2):
        for j in range(1, first.shape[1] - 2):
            # Get the "neighborhood" of 9 pixels centered on our current pixel, for each of the matrices.
            x = j - 1
            y = i - 1
            Ix = gx[y:y + 3, x:x + 3]
            Iy = gy[y:y + 3, x:x + 3]
            It = dt[y:y + 3, x:x + 3]

            # Now do the linear algebra to calculate the Ordinary Least Square regression.
            # Build the matrix A.
            a = np.sum(Ix * Ix)
            b = c = np.sum(Ix * Iy)
            d = np.sum(Iy * Iy)
            A = [[a, b], [c, d]]
            det = a * d - b * c
            if abs(det) < epsilon:
                out[y, x] = np.array([128, 128, 0])
            else:
                AInv = (1 / det) * np.array([[d, -b], [-c, a]])
                # Double-check my math using numpy. It doesn't come out exactly the same, but it seems reasonable?
                check = np.linalg.inv(A)
                if np.abs(np.max(AInv - check)) > 0.005:
                    raise 'The linear algebra is wrong'

                # bee not to be confused with b.
                bee = np.array([[-np.sum(Ix * It)], [-np.sum(Iy * It)]])
                V = AInv @ bee
                Vx = V[0, 0]
                Vy = V[1, 0]
                # Scale and adjust colors to make the graphics work.
                # Vx is blue, Vy is green, magnitude is red. Multiply them all by 50 so smaller changes show up.
                # This does make it truncate larger values, but the image looks decent.
                out[y, x] = np.array([Vx * 50 + 128, Vy * 50 + 128, np.sqrt(Vx ** 2 + Vy ** 2) * 50])

    # Output results.
    print(f"Vx range: {np.min(out[:, :, 0])} to {np.max(out[:, :, 0])}")
    print(f"Vy range: {np.min(out[:, :, 1])} to {np.max(out[:, :, 1])}")
    print(f"Magnitude range: {np.min(out[:, :, 2])} to {np.max(out[:, :, 2])}")
    out = out.astype(np.uint8)
    # Write out the image with all three channels.
    cv2.imwrite(outname + '.jpg', out)

    # Now just blue (Vx).
    z = np.zeros_like(out)
    z[:, :, 0] = out[:, :, 0]
    cv2.imwrite(outname + '_x' + '.jpg', z)

    # Now just green (Vy).
    z = np.zeros_like(out)
    z[:, :, 1] = out[:, :, 1]
    cv2.imwrite(outname + '_y' + '.jpg', z)

    # Now just red (magnitude).
    z = np.zeros_like(out)
    z[:, :, 2] = out[:, :, 2]
    cv2.imwrite(outname + '_z' + '.jpg', z)

d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog2/'
# Read all the images as grayscale. But then convert them to floats so we can mess around with their values
# without getting integer overflow or chopping off decimals.
img1a = cv2.imread(d + 'frame1_a.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img1b = cv2.imread(d + 'frame1_b.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2a = cv2.imread(d + 'frame2_a.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2b = cv2.imread(d + 'frame2_b.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Calculate and plot the optical flows.
optical_flow(img1a, img1b, d + 'of1')
optical_flow(img2a, img2b, d + 'of2')
