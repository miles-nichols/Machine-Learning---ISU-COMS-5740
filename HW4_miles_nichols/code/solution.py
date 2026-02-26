from sklearn.svm import SVC


def svm_with_diff_c(train_label, train_data, test_label, test_data):
    '''
    Use different value of cost c to train a svm model. Then apply the trained model
    on testing label and data.
    
    The value of cost c you need to try is listing as follow:
    c = [0.01, 0.1, 1, 2, 3, 5]
    Please set kernel to 'linear' and keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE

    #create svm with each different C value
    for c in [0.01, 0.1, 1, 2, 3, 5]:
        model = SVC(kernel='linear', C=c)
        model.fit(train_data, train_label)
        predictions = model.predict(test_data)

        # after training and testing, compute accuracy
        correct_predictions = 0
        total_predictions = len(test_label)

        for l in range(total_predictions):
            if predictions[l] == test_label[l]:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Cost value: {c}")
        print(f"Accuracy: {accuracy * 100}%")
        print(f"Total support vectors: {sum(model.n_support_)}")
        print(f"\n")
    


    ### END YOUR CODE
    

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    '''
    Use different kernel to train a svm model. Then apply the trained model
    on testing label and data.
    
    The kernel you need to try is listing as follow:
    'linear': linear kernel
    'poly': polynomial kernel
    'rbf': radial basis function kernel
    Please keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE


    for k in ['linear', 'poly', 'rbf']:
        model = SVC(kernel = k)
        model.fit(train_data, train_label)
        predictions = model.predict(test_data)

        correct_predictions = 0
        total_predictions = len(test_label)

        for l in range(total_predictions):
            if predictions[l] == test_label[l]:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Kernel: {k}")
        print(f"Accuracy: {accuracy * 100}%")
        print(f"Total support vectors: {sum(model.n_support_)}")
        print(f"\n")

    ### END YOUR CODE
