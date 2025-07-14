def linear_cost(x,y,theta,lambda_):
  m = len(y)
  h = x@theta
  cost = 1/(2*m) * np.sum((h-y)**2)
  reg_cost = 1/(2*m)  * np.sum(lambda_* theta[1:]**2)
  return cost+ reg_cost
def gd(x,y,theta,lambda_,lr,iter):
  m=len(y)
  for i in range( iter ):
      
    h=x@theta
    cost = 1/(2*m) *np.sum((h-y)**2)
    gd = 1/m * x.T @ (h-y)
    
    gd[1:]+=(lambda_/m)*theta[1:]
    theta[1:] -= gd[1:]*lr
    print(f'for iter {i},   cost :{cost}')
  
  return theta
  
