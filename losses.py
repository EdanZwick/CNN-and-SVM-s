
import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.
        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======

        indexes = y.repeat(x_scores.shape[1],1) # expand the lables vector to a matrix
        indexes = torch.transpose(indexes,0,1) # transpose the prediction matrix to match scores matrix
        label_scores = torch.gather(x_scores, dim=1, index=indexes) # get the score for each prediction
        M = self.delta + x_scores - label_scores  # now cell (i,j) is margin loss for sample i and class j
        rel = torch.nn.ReLU()
        M = rel(M) # equivalent to max(0,x) defined in function
        sums = torch.sum(M,1)  # sums(i) is now Li(W) (except we have an extra d becuse we included the choosen lable
        loss = torch.mean(sums) - self.delta  # loss is without Norm of chosen W

        # raise NotImplementedError()
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======

        self.grad_ctx = M, x, indexes
        #raise NotImplementedError()
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======

        # idea of this section:
        # By equations given in assigment we know that the gradient is a (D+1,C) matrix (we shall call it G).
        # By the same equations we learn that the j'th col' in G is the mean of all samples who have a positive loss value
        #       with the j'th weight vector (we begin by disregarding the special case for samples who belong to class j).
        #       **this sound a bit confusing but the equations should be clear after thinking about it**
        # We can create a Matrix in the form of G by transposing the sample matrix (x) and multiplying it with a binary
        #       version of M (if the loss value is possitive this matrix will hold 1).
        #       **this is just data manipulation, think about what the cell (i,j) in G hold now and see that it's
        #       exactly what we wanted**
        # We are close but we still didn't account for the fact that for each label (col in G) we need to subtract
        # a vector corresponding to samples who belong to that label.
        # for each sample: it is easy to see from the binary matrix M, how many positive loss values it has (with how many
        # weight vectors did he have a positive value). we'll call this number Pi (P for positive, is sample index).
        # To get the final value of G, for each sample xi that belongs to label "j" we need to subtract (Pi*xi) from col j.
        # Again we can use matrix manipulation to get a matrix holding the values to subtract with matrix manipulation
        # after subtracting that- we just need to devide each coordinate by N and we get the final value.


        M , x , indexes = self.grad_ctx # get needed data from loss function

        Xt = torch.transpose(x, 0, 1)

        mask = torch.ones(M.shape[0],M.shape[1]) # auxilary: tensor.where demands a tensor, we'll use this one
        # turn matrix binary - every cell larger than 0 turns to 1 (since we didn't have negatives we are left with 0's)
        M = torch.where(M > 0, mask, M)


        grad = torch.matmul(Xt,M) # Intermediate result - we still need to subtract values and normalize by N

        sums = torch.sum(M, 1) # number of positive margin losses for a sample - Pi
        Xt = Xt * sums.view(-1) # this is a manipulation to multiply each col j in Xt by scalar sums[j]

        #ybig = y.repeat(M.shape[1],1) # this is the true class vector expanded so can be used in scatter
        #ybig = torch.transpose(ybig, 0, 1) #transposed to be correct dim's for scatter
        # we preform scatter on tmp to get a matrix where tmp(i,j)=1 iff sample i belongs to calss j
        tmp = torch.zeros(M.shape[0], M.shape[1]).scatter_(1, indexes, 1)
        tmp = torch.matmul(Xt,tmp) #now tmp is exactly the amount to subtract from G to get final answer
        grad = grad - tmp
        grad = grad / x.shape[0] #normalize and finished! finaly going to sleep


        #raise NotImplementedError()
        # ========================

        return grad
