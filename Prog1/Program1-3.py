# CS 510 Comp Vision
# Programming assignment 1.3 / Ness Blackbird
# SIFT.

import cv2
# Keep the overall number of keypoints down to a dull roar. Otherwise, we'll end up with a comparison
# matrix of >250M.
sift = cv2.SIFT_create(nfeatures=1000)
import numpy as np

d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/'
# Read the images in BGR format.
img1 = cv2.imread(d + 'SIFT1_img.jpg')
img2 = cv2.imread(d + 'SIFT2_img.jpg')

# Determine keypoints. It works better with grayscale, we don't need color information with this algorithm, which
# is shape-oriented.
keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
# Pick just the top 20% of keypoints. We have too many.
keypoints1 = sorted(keypoints1, key=lambda x: int(x.response), reverse=True)
keypoints2 = sorted(keypoints2, key=lambda x: int(x.response), reverse=True)

img_with_keypoints1 = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_keypoints2 = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(d + 'SIFT_1_with_points.jpg', img_with_keypoints1)
cv2.imwrite(d + 'SIFT_2_with_points.jpg', img_with_keypoints2)

# Calculate the nearest neighbor for each descriptor1.
neighbors_matrix = np.dot(descriptors1, descriptors2.T)
# Find the best matches.
best_matches = np.argmax(neighbors_matrix, axis=1)[:]

# Now get the points indicated in each match. I can't use numpy here, keypoints are lists.
pts = [[best_matches[i], keypoints1[i], keypoints2[i]] for i in best_matches]

# Sort the points by the distance between them, and select only the best 10%.
pts = sorted(pts, key=lambda x: x[0], reverse=True)[0: int(len(pts) * 0.1)]

cv2.destroyAllWindows()