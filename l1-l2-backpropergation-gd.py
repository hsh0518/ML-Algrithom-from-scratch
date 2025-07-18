from  sklearn.linear_model import Lasso,Ridge
lasso = Lasso(alpha = 0.1)
ridge = Ridge(alpha=0.1)

lasso.fit(X_train, y_train)
right.fit(X_train,y_train)

import numpy as np
#L2 ridge
def compute_cost(X,y,theta,lambda_):
   '''
     theta: wights in linear regreassion
     lambda: strength of the regulation, the bigger the stronger of the regulation
     X: we assume it's already handled bias, by adding 1s in front of the feature matrix
     x.dot(theta): feature matrix * weights array to generate prediction
     why 1/2*m: the extra 1/2 is will be cancled out at the time of gradiant decient.
                 the derivetive will be looking cleaner: with cost **@, the derivitive will have a 2 in front of the derivative, 1/2 will cancel it out
   '''
   m=len(y)
   prediction = X.dot(theta)
   cost= (1/2*m) * np.sum((prediction - y)**2)
   reg_cost = (lambda_/2*m) * np.sum(thetap[1 :]**2)
   return cost+reg_cost
# Lasso
def compute_costL1(X,y,theta,lambda_):
    m=len(y)
    prediction = X.dot(theta)
    cost = (1/2*m)*np.sum( (prediction -y)**2)
    reg_cost = (lambda_)/2*m * np.sum(abs(theta[1:]))
    return cost+reg_cost
#grediant descent
def gradient(X,y,theta, alpha, lambda_, num_iters):
    '''
    MSE(loss) : errors *(errors)T and sum it up
    gradient desent: slop of loss respeact to theta_n, output is a
    alpha: learning rate
    lambda：regulation power
    theta: weights
    num_iteration: #updates w
    

    '''
    for _ in len(num_iters):
        m=len(y)
        prediction = X.dot(theta)
        errors = prediction-y
        gradient = (1/m) * X.T.dot(errors)
        #adding the regulation part to the gradient desent
        gradient[1:]+=lambda_/m * theta[1:]
        theta -=lr*gradient
    return theta
  #pass data into dataloader: split and shuffle data into minibatches
from torch.utils.data import DataLoader, TensorDataset

# sample data
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([0.0, 1.0])

dataset = TensorDataset(X,y)  #X and y have to by torch tensors
dataloader = DataLoader(dataset,batch_size=16,shuffle=True) #16:number of samples. it create a 2 component batches, data and target




import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10,1) #linear model:10 feature,1output. randomly assign the initial weights to all features
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01, weight_decay = 1e-4)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output,target)
    loss.backward(    )
    optimizer.step()


