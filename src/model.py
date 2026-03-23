import numpy as np

def one_hot(Y):
        one_hot_Y = np.zeros(10, Y.size)
        for i in range(Y.size[1]):
            one_hot_Y[Y[0, i], i] = 1
        return one_hot_Y

    # One-hot encoding, is a technique used to convert categorical labels into a binary matrix representation.
    # Why we have to do this? Because the output layer of our neural network cannot distinguish with labels like us humans, it can only process numerical data. 
    # By converting the labels into a one-hot encoded format, we can represent each class as a binary vector, where the index corresponding to the class is set to 1 and all other indices are set to 0. 
    # This allows the neural network to learn and make predictions based on the encoded labels effectively.

def init_param():
        # Initialize weights and biases
        W1 = np.random.rand(10, 784) - 0.5 # Weight for hidden layer 1, 10 neurons, 784 inputs, a matrix of shape (10, 784)
        b1 = np.random.rand(10, 1) - 0.5   # Bias for hidden layer 1
        # Weights and biases for output layer
        W2 = np.random.rand(10, 10) - 0.5  # Weight for output layer, 10 neurons, 10 inputs (from hidden layer), a matrix of shape (10, 10)
        b2 = np.random.rand(10, 1) - 0.5   # Bias for output layer
        return W1, b1, W2, b2
    
def ReLU(Z):
        return np.maximum(0, Z)
    ## ReLU activation function, activation function is used to introduce non-linearity into the model, allowing it to learn complex patterns in the data. 
    # It is defined as f(x) = max(0, x), which means that it outputs the input directly if it is positive; otherwise, it outputs zero. 
    # This helps the model to capture non-linear relationships between the input features and the target variable.
    
def softmax(Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ/np.sum(expZ, axis = 0)
    ## Softmax activation function, is used in the output layer of a neural network for multi-class classification problems.
    # It converts the raw output scores (logits) from the output layer into probabilities that sum to 1.
    # The softmax function is defined as f(x_i) = exp(x_i) / sum(exp(x_j)) for all j, where x_i is the input to the function and the sum is taken over all inputs.
    # This allows the model to output a probability distribution over the classes, which can be used to make predictions by selecting the class with the highest probability.

def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    ## Forward propagation, is the process of passing the input data through the layers of the neural network to compute the output predictions.
    # It involves calculating the weighted sum of the inputs and applying the activation function at each layer to produce the output of that layer, which then serves as the input for the next layer.
    # In this function, we compute the linear transformation for the hidden layer (Z1), apply the ReLU activation function to get A1, then compute the linear transformation for the output layer (Z2) and apply the softmax activation function to get the final output probabilities (A2).
    # The output A2 can then be used to make predictions by selecting the class with the highest probability.



