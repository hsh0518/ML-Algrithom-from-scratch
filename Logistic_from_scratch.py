def sigmoid(x,theta):
  z = x@theta
  return 1/(1+np.exp(-z))
def cost(x,y,theta,lambda_):
  m = len(y)
  h=sigmoid(x,theta)
  cost = -1/m *(y.T@np.log(h) +(1-y).T@(np.log(1-h)))
  cost+= (lambda_/(2*m))  * np.sum( theta[1:]**2)
  return cost
def gd(x,y,theta,lr,lambda_,iter):
  m = len(y)
  for i in range(iter):      
    h=sigmoid(x,theta)
    c = cost(x,y,theta,lambda_)
    gd = (1/m) * x.T@(h-y)
    gd[1:]+= (lambda_/m) * theta[1:]
    theta  -= lr*gd 
    print(f'for iteration {i}: cost is {c}')
  return theta
    
  
    
  

