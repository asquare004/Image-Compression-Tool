import sys
import numpy as np
from skimage import io
from sklearn.cluster import KMeans

def compressImage(image, clusters):
  rows, columns, channels = image.shape[0], image.shape[1], image.shape[2]
  image = np.reshape(image, (rows*columns, channels))
  algorithm = KMeans(n_clusters = clusters)
  algorithm.fit(image)
  centroids = np.asarray(algorithm.cluster_centers_, dtype = np.uint8)
  labels = np.asarray(algorithm.labels_, dtype = np.uint8)
  labels = np.reshape(labels, (rows, columns))
  compressedImage = np.zeros((rows, columns, channels), dtype = np.uint8)
  for i in range(rows):
    for j in range(columns):
        compressedImage[i, j, :] = centroids[labels[i, j], :]
  return compressedImage

filename = sys.argv[1]
image = io.imread(filename)
clusters = int(input("Enter the number of clusters : "))
compressedImage = compressImage(image, clusters)
io.imsave(filename.split(".")[0]+"-compressed.png", compressedImage)