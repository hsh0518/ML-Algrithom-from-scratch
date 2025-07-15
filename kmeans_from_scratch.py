class Kmeans():
  def __init__(self, k, tol, iter):
    self.k = k
    self.tol = tol
    self.iter=iter
    self.centroids = None
    self.labels = None
  def _calc_dist(self, x):
    return np.linalg.norm (x[:,np.newaxis]-self.centroids,axis =2) # new axis will hold the distance to each centriod. axis=2 will remove the feature vectors from table and only return samples and it's distance to the centroids
  def train(self, x):
    r, c = x.shape
    randomindex= np.random.choice(r,self.k,replace=False)
    self.centroids = x[randomindex]
    for i in range(self.iter):
      dist = self._calc_dist(x)
      labels = np.argmin(dist,axis=1)
      new_c = np.array([x[labels==j].mean(axis=0) if np.any (x[labels==j] )else self.centroids[j] for j in range(self.k)])
      diff = np.linalg.norm(new_c-self.centroids)
      if diff<self.tol:
        break
      self.centroids = new_c
    self.labels=labels
    return self.centroids, self.labels

    
