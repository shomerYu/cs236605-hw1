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
        self.weights = torch.randn((self.n_features,self.n_classes))*weight_std

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
        class_scores = x.mm(self.weights)
        y_pred = torch.argmax(class_scores, 1)
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
        acc = float(torch.sum(y == y_pred))/len(y)
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
            len_train = len(dl_train)
            len_valid = len(dl_valid)
            mean_acc = 0
            mean_loss = 0
            
            mean_grad = torch.zeros(self.weights.shape)
            for (x, y) in dl_train:

                y_pred, class_score = self.predict(x)
                mean_acc += self.evaluate_accuracy(y, y_pred)
                mean_loss += loss_fn.loss(x, y, class_score, y_pred)
                mean_grad += loss_fn.grad()

            mean_acc /= len_train
            mean_loss /= len_train

            train_res.accuracy.append(mean_acc)
            train_res.loss.append(mean_loss)

            mean_acc2 = 0
            mean_loss2 = 0
            for (x2, y2) in dl_valid:
                y_pred, class_score = self.predict(x2)
                mean_acc2 += self.evaluate_accuracy(y2, y_pred)
                mean_loss2 += loss_fn.loss(x2, y2, class_score, y_pred)

            mean_acc2 /= len_valid
            mean_loss2 /= len_valid

            self.weights -= learn_rate * mean_grad / len_train

            valid_res.accuracy.append(mean_acc2)
            valid_res.loss.append(mean_loss2)
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
        w = self.weights

        if has_bias:
            w = w[:-1, :]
        aa = torch.Tensor(self.n_classes, img_shape[0], img_shape[1], img_shape[2])
        # aa = torch.Tensor(self.n_classes, for x in img_shape: x)
        w_images = torch.zeros(aa.shape)
        for i in range(self.n_classes):
            w_images[i] = w[:, i].reshape(aa.shape[1:])

        # ========================

        return w_images
