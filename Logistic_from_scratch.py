def sigmoid(x,theta):
  z = x@theta
  return 1/(1+np.exp(-z))
def cost(x,y,theta,lambda_):
  m = len(y)
  h=sigmoid(x,theta)
  cost = -1/m *(y.T@np.log(h) +(1-y).T@(np.log(1-h)))
  cost+= (lambda_/(2*m))  * np.sum( theta[1:]**2)
  return cost
def gd(X_train, y_train, X_val, y_val, theta, lambda_, lr, max_iters, patience):
    best_theta = theta.copy()
    best_val_loss = float('inf')
    no_improve = 0

    for i in range(max_iters):
        # Training step
        h = sigmoid(X_train, theta)
        grad = (1/len(y_train)) * (X_train.T @ (h - y_train))
        grad[1:] += (lambda_ / len(y_train)) * theta[1:]
        theta -= lr * grad

        # 计算训练 & 验证集 loss
        train_loss = compute_loss(X_train, y_train, theta, lambda_)
        val_loss = compute_loss(X_val, y_val, theta, lambda_)

        print(f"Iter {i}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # early stopping 判断
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_theta = theta.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at iter {i}. best_val_loss = {best_val_loss:.4f}")
                break

    return best_theta
    
  
    
  

