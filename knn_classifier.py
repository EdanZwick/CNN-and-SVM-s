import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import helpers.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            indexes = torch.topk(dist_matrix[:, i], self.k, largest = False)[1] #get indexes for k nearest neighbours
            labels = torch.gather(self.y_train,dim=-1,index=indexes) #get classes for each of the neighbours
            y_pred[i] = torch.mode(labels)[0].item() #what label is most common
            # ========================
        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        dists = torch.tensor([])
        # ====== YOUR CODE: ======

        tsttrans = torch.transpose(x_test, 0, 1) #Transposition will allow matrix multification so that we will get output of wanted size
        prod = torch.matmul(self.x_train, tsttrans, out=None) #multiply train^-1 * test, matrix is of size (Ntrain,Ntest)
        prod = prod * (-2) # product by -2

        sq_test = x_test**2 # squares the tensor elementwise
        sq_train = self.x_train**2

        sumtest = torch.sum(sq_test, 1) #sums the squared elements of the matrix by row, vector of Ntest
        sumtrain = torch.sum(sq_train, 1) # vector of Ntrain


        test_big = sumtest.repeat(len(sumtrain), 1) #extends the sum vector to create a sum matrix of: (NtrainXNtest)
        train_big = sumtrain.repeat(len(sumtest), 1) #extends the sum vector to create a sum matrix of: (NtestXNtrain)

        train_big = torch.transpose(train_big, 0, 1)

        torch.sqrt(prod + train_big + test_big, out=dists)

        #raise NotImplementedError()
        # ========================

        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    compares = torch.eq(y, y_pred)
    n = compares.shape[0]
    s = torch.nonzero(compares).shape[0]
    accuracy = s / n
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []
    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be aS
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        k_accuracies = []
        for j in range(num_folds):
            training, validation = \
                dataloaders.create_train_validation_loaders(ds_train, 1/num_folds)
            validation_tensor, y = dataloader_utils.flatten(validation)
            model.train(training)
            y_pred = model.predict(validation_tensor)
            k_accuracies.append(accuracy(y, y_pred))

        accuracies.append(k_accuracies)

        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
