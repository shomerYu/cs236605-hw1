r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing K improves generalization since if we take into account more neighbours we can make the result less 
vulnerable to outliers.
Cross validation is as seen in the exercise used to find the best k.
There are two edge cases when K = N then we will take into account all of the samples and thus simply return the most common class disregarding the real value of the sample, or when K = 1 and thus the class of the sample will be the class of the closest neighbour,which could have an error in his prediction.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The selection of $\Delta > is arbitrary for the SVM because of the L(w) derivation, the optimal solution does not
depend on  $\Delta hence it is an arbitrary choice. 

"""

part3_q2 = r"""
**Your answer:**

1.
the linear model "learn" dominant features in each number that are common among most
of the given training set, for example it's easy to see in the weights representing the digit 0, 
that the model learned a round shape, similar to zero.
the classification error are from digits that have features similar to other digits, for example the digit 4
can be mistaken be the number 9 in some cases.

2.
the KNN model looks at the K most similar images by looking at each pixel at a time while the SVM loss 
 model only looks at the images which have an effect on the separating hyper-plane. 
"""

part3_q3 = r"""
**Your answer:**


1.
from looking at the training loss and accuracy we can tell that the learning rate is good. at the beginning there
are some big jumps at the accuracy and small jumps at the loss but the overall behavior of the two graphs good.
 if we would decrease the training rate the training loss would be much higher than it is for this learning rate, 
 which means that it will take more training time to reach the same results. for higher learning rate the learning 
 could become "unstable" and could not converge.  

2.
from looking at the training and test results we can say that the training Slightly overfitted to the training set.
the training and validation results at the end of the training is around 85%-90% and the test set is around 80%, this 
is actually not so bad results because we will always have some over-fitting. 

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

the perfect behavior we expect in the residual plot is a perfect monotonous decreasing or increasing points
that "sits" separate from each other on a straight line.
that kind of behavior means that for every given data (x axis) we can get one price estimation. (Injective function
)


"""

part4_q2 = r"""
**Your answer:**

1. The use of the 'np.logspace' function, as opposed to the 'np.linspace' function, is allowing us to search over
 lamdas in different orders of magnitudes. Essentially it is sort of like a "linspace for the order of magnitude of
$\lambda$"

2. Overall, the model is fitted len(degree_range) * len(lamda_range) for all the parameters, and for each
set of parameters K_folds so in totel len(degree_range) * len(lamda_range)*K_folds
(and finaly the final fit over the entire training data)
"""

# ==============
