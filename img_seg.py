import numpy as np
import numpy.linalg as linalg
import operator

# INPUTS:
# X - n x p 2d array; where n is the number of data points, p is the dimension
# k - number of clusters 
# cluster_assignment - initial cluster assignment (range from 1 to k); this is a 1d array of size n
# maxIters - number of iterations to run the Lloyd's algorithm 
# OUTPUTS:
# cluster_assignment - final cluster assignments
# centroids - k x p 2d array; where i^th row contains the center for i^th cluster
# Note that the order of output matters; use the following order: cluster_assignment, centroids

    
def kmeans(X, k, cluster_assignment, maxIters):
    centers = [[0.0, 0.0, 0.0]]*k
##    print(centers)
##    print(len(X), k, maxIters)
##    print(X[0])
##    print(cluster_assignment[0:100])
##    maxIters = 20
    for i in range(maxIters):
        print("in iteration:", i)
        for j in range(k):
##            print("in iteration:", i, " cluster:", j)
            subsetX = []
            subsetY = []
            subsetZ = []
            idx = np.squeeze(np.where(cluster_assignment == j + 1))
##            print(type(idx))
##            print()
            try:
                length = len(idx)
                if (len(idx)==0):
##                    print("   cluster", j+1, "is empty")
                    centers[j] = [float('Inf'), float('Inf') ,float('Inf')]
                    continue
            except:
##                print("   cluster", j+1, "is empty")
                centers[j] = [float('Inf'), float('Inf') ,float('Inf')]
                continue
            
##            print ("   cluster ", j+1, "has size:", len(idx))
            n = len(idx)
            
            for g in idx:
                subsetX.append(X[g][0])
                subsetY.append(X[g][1])
                subsetZ.append(X[g][2])
##                print(g)
            centers[j] = [np.mean(subsetX), np.mean(subsetY), np.mean(subsetZ)]
##            print("      centering at:", centers[j])


        distances = []
        for h in range(len(centers)):

            D = np.sum((X - centers[h])**2, axis=1)
##            print(len(D))
            distances.append(D)
##            print(len(distances))
        cluster_assignment = np.argmin(distances, axis=0) + 1

            
##        for l in range(len(X)):
##            distance = float('Inf')
##            index = 0
##            for h in range(len(centers)):
##                if centers[h][0] == float('Inf'):
##                    continue
##                x = X[l]
##                d = dist(x, centers[h])
##                if d < distance:
##                    index = h
##                    distance = d
##            cluster_assignment[l] = index + 1
##        print(cluster_assignment)

    return (cluster_assignment, centers)    
        

		
	
