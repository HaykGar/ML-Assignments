import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)

    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out = np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            X_out = np.hstack((np.ones((N, 1)), X))
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            
            # X_out = np.hstack(  (np.ones((N, 1)), X,  np.hstack([X**n for n in range(2, m+1)]) ) )
            X_out = np.hstack([X**n for n in range(m+1)])  
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w)
        m = self.m
        N,d = X.shape 
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #     

            X = self.gen_poly_features(X)
            
            error = (X @ self.w).reshape(-1, 1) - y.reshape(-1, 1)
            loss += ( (self.reg / 2)  *  np.sum((self.w[1:]) ** 2)  +  (np.sum(error**2) ) ) / N
            
            reg_grad =  np.concatenate( (np.zeros((1, 1)), self.reg * self.w[1:]) ) / N
            g = 2 * ( X.T @ error ) / N
            grad +=   g + reg_grad
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
            
            X = self.gen_poly_features(X)
            
            error = (X @ self.w).reshape(-1, 1) - y.reshape(-1, 1)
            loss += ( (self.reg /2)  *  np.sum((self.w[1:]) ** 2)  +  (np.sum(error**2) ) ) / N
            
            reg_grad =  np.concatenate( (np.zeros((1, 1)), self.reg * self.w[1:]) ) / N
            g = 2 * ( X.T @ error ) / N
            grad +=   g + reg_grad
            
            # error = self.predict(X) - y
            
            # X = self.gen_poly_features(X)
            
            # loss += np.linalg.norm(error, ord = 2)**2 / N
            
            # grad += (self.reg * self.w) +  (2 / N) * np.dot(X.T, error.reshape((-1,1)))
            
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        
        mean = X.mean(axis = 0)
        std_dev = X.std(axis = 0)
        
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
                
                indices_chosen = np.random.choice(N, batch_size, replace = False)
                X_batch = X[indices_chosen]
                y_batch = y[indices_chosen]
                
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss = 0.0
                grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
            
                X_batch = (X_batch - mean) / (std_dev + 1e-7)
                
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w -= eta * grad
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #
            
            X1 = self.gen_poly_features(X)
            
            xtx_inv = np.linalg.inv(np.dot(X1.T, X1))
            self.w = (xtx_inv @ X1.T @ y).reshape(-1, 1)
                        
            loss, _ = self.loss_and_grad(X, y)
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
                
            X1 = self.gen_poly_features(X)
            
            xtx_inv = np.linalg.inv(np.dot(X1.T, X1))
            self.w = (xtx_inv @ X1.T @ y).reshape(-1, 1)
                        
            loss, _ = self.loss_and_grad(X, y)
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m        
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            X = self.gen_poly_features(X)
            mean = X.mean(axis = 0)
            std_dev = X.std(axis = 0)
            X = (X - mean) / (std_dev + 1e-7)
            
            y_pred += (X @ self.w).reshape(-1)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X = self.gen_poly_features(X)
            mean = X.mean(axis = 0)
            std_dev = X.std(axis = 0)
            X = (X - mean) / (std_dev + 1e-7)
            
            y_pred += (X @ self.w).reshape(-1)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred