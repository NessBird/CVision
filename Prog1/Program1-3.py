# CS 510 Comp Vision
# Programming assignment 1.3 / Ness Blackbird
# SIFT.

import cv2
# Keep the overall number of keypoints down to a dull roar. Otherwise, we'll end up with a comparison
# matrix of >250M.
import numpy as np

d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog1/'
# Read the images in BGR format.
img1 = cv2.imread(d + 'test1.jpg')
img2 = cv2.imread(d + 'test2.jpg')
# Make them the same size.
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

sift = cv2.SIFT_create(nfeatures=1000)

# Determine keypoints. It works better with grayscale, we don't need color information with this algorithm, which
# is shape-oriented.
keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

# However, we save the original image for display.
img_with_keypoints1 = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_keypoints2 = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(d + 'SIFT_1_with_points.jpg', img_with_keypoints1)
cv2.imwrite(d + 'SIFT_2_with_points.jpg', img_with_keypoints2)

# Calculate the nearest neighbor for each descriptor1.
neighbors_matrix = np.sqrt(np.sum((descriptors1[:, np.newaxis] - descriptors2) ** 2, axis=2))
# Find the best matches, one for each descriptors1.
best_matches = np.argmin(neighbors_matrix, axis=1)
# Stay in numpy world. Add in the index into keypoints1.
match_values = neighbors_matrix[np.arange(len(best_matches)), best_matches]
# Build a sort index, sorted by the distance, indexing into keypoints.
sort_index = np.argsort(match_values)
# Chop off the bottom 90%.
sort_index = sort_index[:sort_index.shape[0] // 10]

# Build DMatch objects for use by cv2.drawMatches().
matches = [cv2.DMatch(sort_index[i], best_matches[sort_index[i]], match_values[sort_index[i]])
           for i in range(len(sort_index))]
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
cv2.imwrite(d + 'SIFT_1_matches.jpg', img_matches)
