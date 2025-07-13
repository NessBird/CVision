# CS 510 Comp Vision
# Programming assignment 1.2(ii) / Ness Blackbird
# Read images and implement K-means to flatten them to 5 or 10 colors.
import numpy as np
import matplotlib.pyplot as plt

# Read the data.
d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/510_cluster_dataset.txt'
f = open(d, "r")

# Build the data as a Python list, it's a lot easier.
dataset = []
datum = f.readline().strip()
while datum:
    # Remove redundant spaces.
    trimmed = " ".join(datum.split())
    # Now split it on the one remaining space.
    trimmed = trimmed.split()
    dataset.append(trimmed)
    datum = f.readline().strip()

# Convert the data to an np array of floating points values.
dataset = np.array(dataset, dtype=float)
# Set up an array just for the classes.
classes = np.zeros(len(dataset))

# We'll try different numbers of clusters. For each, we'll want to record the sum of squares error.
k = [2, 3, 4]

# Try each number several times.
iterations = 25

convergence_check = 0.00001

for k_idx, i in enumerate(k):
    # Run it several times to find the best fit.
    best_sos = np.inf
    for j in range(iterations):
        # Select k random data points, and run the model from there.
        centers = dataset[np.random.choice(len(dataset), i, replace=False)]
        # Run the algorithm until convergence.
        while True:
            old_centers = centers.copy()
            # Calculate distances from all points to all centers. This makes a matrix with shape
            # (n_points, n_clusters). Along the way, it generates a tensor of shape (n_points, n_clusters, n_features),
            # then sums along the features axis.
            distances = np.sqrt(((dataset[:, np.newaxis, :] - centers[np.newaxis, :, :])**2).sum(axis=2))

            # Find the closest cluster for each point, and assign it to that cluster.
            classes = np.argmin(distances, axis=1)

            # Move the cluster centers to the middle of each cluster.
            for c in range(len(centers)):
                # Collect up the points in this cluster.
                cluster_points = dataset[classes == c]
                # This will return nan if the cluster has no points.
                centers[c] = np.mean(cluster_points, axis=0)

            # Check for convergence.
            divergence = np.sum((centers - old_centers) ** 2)
            if divergence < convergence_check:
                # Calculate the sum of squares distance per cluster.
                sos = np.sum(distances[np.arange(len(classes)), classes]**2)
                if sos < best_sos:
                    best_sos = sos
                    best_centers = centers.copy()
                    best_classes = classes.copy()
                break

    plt.figure(figsize=(10, 8))

    # Plot each data point colored by cluster
    plt.scatter(dataset[:, 0], dataset[:, 1], c=best_classes, cmap='rainbow',
                alpha=0.7, s=50, edgecolors='w')

    # Plot the cluster centers as larger points
    plt.scatter(best_centers[:, 0], best_centers[:, 1], c='red', s=200, alpha=0.9,
                marker='X', edgecolors='black', label='Cluster Centers')

    plt.title(f'K-means Clustering Results, {i} clusters, SoS: {best_sos:.0f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(d + "K-means-" + str(i) + ".png", dpi=300)
