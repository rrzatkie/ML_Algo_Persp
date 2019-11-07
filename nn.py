import numpy as np

def train(inputs, targets, weights, eta, niter):
    
    for i in range(niter):

        activations = inputs.dot(weights)
        activations = np.where(activations>0, 1, 0)

        weights -= eta*np.dot(np.transpose(inputs), activations-targets)
    return weights

def main():
    X = np.array([[1, 0],
                [0, 0],
                [1, 1],
                [0, 1]])

    T = np.array([1,0,1,1])

    W = np.array([-0.02, 0.02])
    print(W)
    
    eta = 0.25

    W = train(X, T, W, eta, 10)

    activations = X.dot(W)
    
    print("Targets: {}".format(T))
    print("Computed: {}".format(np.where(activations>0, 1, 0)))

    print(W)

main()