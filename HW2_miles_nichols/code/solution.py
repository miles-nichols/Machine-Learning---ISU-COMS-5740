import numpy as np
import sys
from helper import *
import matplotlib.pyplot as plt
import numpy as np  


# Our model is a linear perceptron that will look at a subset of the MNIST datset and distinguish between a handwritten 1 and 5.
# The first entry of the data is digit's current label (1 or 5). The next 256 are grayscale values between -1 and 1 that correspond to a 16by16 image.
# This model will make its decision based two features: symmetry and average intensity. 
# helper.py loads raw text fiiles and reshapes them into 16*16 image matricies. It calculates symmetry and intensity for each image 
# The code computes a 2D feature (symmetry and intensity) from each image. So each digit image is represented by a 2D vector before being augmented with a 1 to form
# a 3D vector as disccussed in class. These features along with the corresponfing labels should serve as inputs to the perceptron algorithm.




def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE

    # (#images, h, w)
    if data is not None and len(data) > 0: # data not empty
        im = [] # new 2d list to hold image
        for d in data:
            im.append(d) # 2d list of 2 images, each image is 16*16

    plt.clf()
    plt.imshow(im[0], cmap='gray')
    plt.savefig('image_1.png')
    plt.clf() #clear plot
    plt.imshow(im[1], cmap='gray')
    plt.savefig('image_2.png')
    


    ### END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    ### YOUR CODE HERE

    # X represent the input data (features) and y represent the labels (answers)
    # x is all of the rows in column 0 (symmetry) and y is the rows in column 1 (intensity)

    groupA_x = []
    groupA_y = []
    groupB_x = []
    groupB_y = []
    i = 0
    for why in y:
        
        if y[i] == 1:
            groupA_x.append(X[i, 0])
            groupA_y.append(X[i, 1])
        else:
            groupB_x.append(X[i, 0])
            groupB_y.append(X[i, 1])
        i = i + 1
            

    plt.clf()
    # Plotting the first group as blue plus signs
    plt.scatter(groupA_x, groupA_y, color='blue', marker='+')

    # Plotting the second group as red stars right on top
    plt.scatter(groupB_x, groupB_y, color='red', marker='*') 
    if save is True:
        plt.savefig('train_features.png')

    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        
        w = [0, 0 , 0] # w0 is 1, w1 is symmetry, w2 is intesity (weights)
        hx = 0

        for loop in range(self.max_iter): #itterate maxiter times
            i = 0
            for image in X: # for each image extract features and calculate h(x)
                symmetry = X[i , 1]
                intensity = X[i, 2]
                hx = w[0] * 1 + w[1] * symmetry + w[2] * intensity
                if(hx > 0):
                    hx = 1
                else:
                    hx = -1
                
                #update (w = w + yi * xi)
                if hx != y[i]:
                    w[0] = w[0] + (y[i] * 1)
                    w[1] = w[1] + (y[i] * symmetry)
                    w[2] = w[2] + (y[i] * intensity)
                i = i + 1
            
            
            

        # After implementation, assign your weights w to self as below:
        self.W = np.array(w)
        
        ### END YOUR CODE
        
        return self

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE

        preds = [] # list of pedictions for each image
        i = 0
        w = self.W
        for image in X:
            symmetry = X[i , 1]
            intensity = X[i, 2]
            hx = w[0] * 1 + w[1] * symmetry + w[2] * intensity
            if(hx > 0):
                hx = 1
            else:
                hx = -1
            preds.append(hx)
            i = i + 1
        return np.array(preds)

        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE
        score = 0
        i = 0
        preds = self.predict(X) # get the predictions into an array
        for p in preds:
            if p == y[i]: # if prediction is correct inc score
                score = score + 1
            i = i + 1
        return score / i # return the average

        ### END YOUR CODE




def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    ### YOUR CODE HERE


    # decision boundary: 0 = w0 + (w1 * symmetry) + (w2 * intesnsity)
    # x2 = - w1/w2 * x1 - w0/w2

    plt.clf() # clear plot
    show_features(X, y, False) # make scatter plot

    # plot boundary 
    # get the largest and smallest value for symmetry
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    x_coords = np.array([x_min, x_max]) 
    
    # 3. plug in w values and x_coords for x1 (y_coords is x2)
    y_coords = -(W[1] / W[2]) * x_coords + -(W[0] / W[2])
    
    plt.plot(x_coords, y_coords, color='black')
    plt.savefig('result.png')

    ### END YOUR CODE



def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()

    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return W, train_acc, test_acc