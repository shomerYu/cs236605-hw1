import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import unittest

#matplotlib inline
#load_ext autoreload
#autoreload 2

plt.rcParams.update({'font.size': 12})
torch.random.manual_seed(1904)
test = unittest.TestCase()

# Prepare data for Linear Classifier
import torchvision.transforms as tvtf
import hw1.datasets as hw1datasets
import hw1.dataloaders as hw1dataloaders
import hw1.transforms as hw1tf

# Define the transforms that should be applied to each image in the dataset before returning it
tf_ds = tvtf.Compose([
    tvtf.ToTensor(), # Convert PIL image to pytorch Tensor
    tvtf.Normalize(
        # Normalize each chanel with precomputed mean and std of the train set
        mean=(0.49139968, 0.48215841, 0.44653091),
        std=(0.24703223,  0.24348513, 0.26158784)),
    hw1tf.TensorView(-1), # Reshape to 1D Tensor
    hw1tf.BiasTrick(), # Apply the bias trick (add bias dimension to data)
])

# Define how much data to load
num_train = 10000
num_test = 1000
batch_size = 1000

# Training dataset
ds_train = hw1datasets.SubsetDataset(
    torchvision.datasets.MNIST(root='./data/mnist/', download=True, train=True, transform=tf_ds),
    num_train)

# Create training & validation sets
dl_train, dl_valid = hw1dataloaders.create_train_validation_loaders(
    ds_train, validation_ratio=0.2, batch_size=batch_size
)

# Test dataset & loader
ds_test = hw1datasets.SubsetDataset(
    torchvision.datasets.MNIST(root='./data/mnist/', download=True, train=False, transform=tf_ds),
    num_test)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size)

x0, y0 = ds_train[0]
n_features = torch.numel(x0)
n_classes = 10

# Make sure samples have bias term added
test.assertEqual(n_features, 28*28*1+1, "Incorrect sample dimension")

import hw1.linear_classifier as hw1linear

# Create a classifier
lin_cls = hw1linear.LinearClassifier(n_features, n_classes)

# Evaluate accuracy on test set
mean_acc = 0
for (x,y) in dl_test:
    y_pred, _ = lin_cls.predict(x)
    mean_acc += lin_cls.evaluate_accuracy(y, y_pred)
mean_acc /= len(dl_test)

print(f"Accuracy: {mean_acc:.1f}%")

import cs236605.dataloader_utils as dl_utils
from hw1.losses import SVMHingeLoss

# Create a hinge-loss function
loss_fn = SVMHingeLoss(delta=1)

# Classify all samples in the test set (because it doesn't depend on initialization)
x, y = dl_utils.flatten(dl_test)
y_pred, x_scores = lin_cls.predict(x)
loss = loss_fn(x, y, x_scores, y_pred)

# Compare to pre-computed expected value as a test
expected_loss = 8.9579
print("loss =", loss.item())
print('diff =', abs(loss.item()-expected_loss))
test.assertAlmostEqual(loss.item(), expected_loss, delta=1e-1)

from hw1.losses import SVMHingeLoss

# Create a hinge-loss function
loss_fn = SVMHingeLoss(delta=1.)

# Compute loss and gradient
loss = loss_fn(x, y, x_scores, y_pred)
grad = loss_fn.grad()

# Test the gradient with a pre-computed expected value
expected_grad = torch.load('tests/assets/part3_expected_grad.pt')
diff = torch.norm(grad - expected_grad)
print('diff =', diff.item())
test.assertAlmostEqual(diff, 0, delta=1e-1)

lin_cls = hw1linear.LinearClassifier(n_features, n_classes)

# Evaluate on the test set
x_test, y_test = dl_utils.flatten(dl_test)
y_test_pred, _ = lin_cls.predict(x_test)
test_acc_before = lin_cls.evaluate_accuracy(y_test, y_test_pred)

# Train the model
svm_loss_fn = SVMHingeLoss()
train_res, valid_res = lin_cls.train(dl_train, dl_valid, svm_loss_fn,
                                     learn_rate=1e-3, weight_decay=0.5,
                                     max_epochs=31)

# Re-evaluate on the test set
y_test_pred, _ = lin_cls.predict(x_test)
test_acc_after = lin_cls.evaluate_accuracy(y_test, y_test_pred)

# Plot loss and accuracy
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for i, loss_acc in enumerate(('loss', 'accuracy')):
    axes[i].plot(getattr(train_res, loss_acc))
    axes[i].plot(getattr(valid_res, loss_acc))
    axes[i].set_title(loss_acc.capitalize(), fontweight='bold')
    axes[i].set_xlabel('Epoch')
    axes[i].legend(('train', 'valid'))
    axes[i].grid(which='both', axis='y')

# Check test set accuracy
print(f'Test-set accuracy before training: {test_acc_before:.1f}%')
print(f'Test-set accuracy after training: {test_acc_after:.1f}%')
test.assertGreaterEqual(test_acc_after, 80.0)