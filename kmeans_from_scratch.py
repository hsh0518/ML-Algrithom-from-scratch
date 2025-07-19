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
      if diff<self.tol: #如果新的中心点（new_c）和旧的中心点（self.centroids）之间的整体移动距离已经很小（小于某个容忍阈值 tol），就可以提前停止迭代，不再继续运行 self.iter 次了
        break
      self.centroids = new_c
    self.labels=labels
    return self.centroids, self.labels

    
