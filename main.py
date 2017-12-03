import numpy as np
import scipy as sp
from scipy import io
from scipy import misc
import img_seg

import matplotlib.pyplot as plt
from PIL import Image


def main():
	# Get data
	img = misc.imread('../data/img1.JPG')
	img = img.astype(np.float32)
	img_shape = np.shape(img)

	# reshape image
	img2 = img.reshape((img_shape[0]*img_shape[1], 3))
	img2 = np.squeeze(img2)
	
	# perform k-means clustering
	k = 200;
	maxIters = 20;
	np.random.seed(24)
	cluster_assignment = np.random.randint(low = 1, high = k+1, size = np.shape(img2)[0])
	cluster_assignment, centroids = img_seg.kmeans(img2, k, cluster_assignment, maxIters);

	# reconstruct the image using learned clusters
	img3 = np.zeros(np.shape(img2));
	for i in range(k):
                
		idx = np.squeeze(np.where(cluster_assignment == i + 1))
##		print("idx",len(idx), idx[0])
		for j in idx:
                        img3[j] = img3[j] + centroids[i];

	img4 = img3.reshape((img_shape[0], img_shape[1], 3))
	img4 = img4.astype(np.uint8)

	# display the image
##	misc.imshow(img4);
	plt.imshow(np.uint8(img4))
	plt.show()
	plt.imshow(np.uint8(img))
	plt.show()
        
        
if __name__ == '__main__':
	main()
