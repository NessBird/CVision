# CS 510 Comp Vision
# Programming assignment 1.2(ii) / Ness Blackbird
# Use K-Means on the colors of the test images, cluster them and display the images with the flattened colors.
import cv2
import numpy as np

def flatten(inp, colors):
    # Return the given image with its colors flattened using K-means to the given number of colors.
    width, height, c = inp.shape
    if c != 3:
        raise ValueError("Expecting 3 channels, found {}".format(c))

    # Turn the image into a list of pixels, temporarily, each with the three RGB attributes.
    image = inp.copy()
    image = image.reshape(-1, 3)
    image = image.astype(np.float32)

    convergence_check = 500
    iterations = 1
    best_sos = np.inf
    for j in range(iterations):
        # Select the number of random data points we need to start with, and run the model from there.
        centers = image[np.random.choice(len(image), colors, replace=False)]

        # Run the algorithm until convergence.
        while True:
            old_centers = centers.copy()
            # Calculate distances from all points to all centers. This makes a matrix with shape
            # (n_points, n_clusters). Along the way, it generates a tensor of shape (n_points, n_clusters, n_features),
            # then sums along the features axis. This is similar to the other one, only it uses 3 color features
            # instead of 2 positional features.
            distances = np.sqrt(((image[:, np.newaxis, :] - centers[np.newaxis, :, :])**2).sum(axis=2))

            # Find the closest cluster for each point, and assign it to that cluster.
            classes = np.argmin(distances, axis=1)

            # Move the cluster centers to the middle of each cluster.
            for c in range(len(centers)):
                # Collect up the points in this cluster.
                cluster_points = image[classes == c]
                # This will return nan if the cluster has no points.
                centers[c] = np.mean(cluster_points, axis=0)

            # Check for convergence.
            divergence = np.sum((centers - old_centers) ** 2)
            if divergence < convergence_check:
                # It has converged. Calculate the sum of squares distance per cluster.
                sos = np.sum(distances[np.arange(len(classes)), classes]**2)
                if sos < best_sos:
                    best_sos = sos
                    best_centers = centers.copy()
                    best_classes = classes.copy()
                break

    # OK, we've found the best version of the k-means. Now we need to make a copy of the image.
    image = best_centers[best_classes]
    image = image.reshape(width, height, 3)
    image = image.astype(np.uint8)
    return image


# Read the images.
d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/'
img1 = cv2.imread(d + 'Kmean_img1.jpg', cv2.IMREAD_COLOR_RGB)
img2 = cv2.imread(d + 'Kmean_img2.jpg', cv2.IMREAD_COLOR_RGB)
if img1 is None or img2 is None:
    raise ValueError("Image not found.")

img = flatten(img1, 3)
cv2.imshow('Image 1, 3 colors', img)
img = flatten(img1, 30)
cv2.imshow('Image 1, 50 colors', img)

img = flatten(img2, 3)
cv2.imshow('Image 2, 3 colors', img)
img = flatten(img2, 30)
cv2.imshow('Image 2, 50 colors', img)

cv2.waitKey(10)
input("Press any key to finish")
cv2.destroyAllWindows()
#When I want to do broadcasting, but I don't need a dimension in the existing data structure, I need to select just the data I need to using : and []. So, for example, I've got an image with 2 features and a dimension for RGB color. The normal way of retrieving an item from that would be color = image[x, y]