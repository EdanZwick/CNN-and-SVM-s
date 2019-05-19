import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        m = torch.zeros(n_features, n_classes) #assuming the "Bias trick" is already accounted in n_features
        self.weights = torch.normal(mean = m, std = weight_std) #just initializing the weight matrix with random values
        #raise NotImplementedError()
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights, out=None) #cell (i,j) -> score of vector i and class j
        y_pred = torch.max(class_scores, 1 )[1] # we take only second returned value: index of max score.

        #raise NotImplementedError()
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        compares = torch.eq(y, y_pred)
        n = compares.shape[0]
        s = torch.nonzero(compares).shape[0]
        acc = s / n
        #raise NotImplementedError()
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======

            num = 0
            i = 0
            for idx, (samples, truth) in enumerate(dl_train):
                predictions, scores = self.predict(samples) #calculate results per current batch
                total_correct += (self.evaluate_accuracy(truth,predictions)) * samples.shape[0]
                average_loss += (loss_fn.loss(samples, truth, scores, predictions)) # loss.grad function uses this also
                average_loss = average_loss + ((weight_decay / 2) * (torch.norm(self.weights)**2))
                #next line will update weight function by formula given in jupyter. Use of weight decay parameter unclear
                self.weights = self.weights - (learn_rate * (loss_fn.grad() + (weight_decay * self.weights)))
                num += samples.shape[0]
                i+=1

            train_res.accuracy.append((total_correct / num))
            train_res.loss.append((average_loss / i))

            total_correct = 0
            average_loss = 0
            num = 0
            i = 0

            for idx, (samples, truth) in enumerate(dl_valid):
                predictions, scores = self.predict(samples) #calculate results per current batch
                total_correct += (self.evaluate_accuracy(truth,predictions)) * samples.shape[0]
                average_loss += (loss_fn.loss(samples, truth, scores, predictions)) # loss.grad function uses this also
                average_loss = average_loss + ((weight_decay / 2) * (torch.norm(self.weights)**2))
                num += samples.shape[0]
                i+=1

            valid_res.accuracy.append(total_correct / num)
            valid_res.loss.append(average_loss / i)

            #raise NotImplementedError()
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res
    
    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        C , H , W = img_shape
        w_images = self.weights
        if has_bias:
            w_images = w_images[:w_images.shape[0]-1,:]
        w_images = w_images.view(w_images.shape[1],C,H,W)
        #raise NotImplementedError()
        # ========================

        return w_images
