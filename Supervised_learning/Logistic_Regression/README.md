# Logistic Regression

This repository showcases the practical implementation and diverse applications of logistic regression.

## File descriptions
Inside "Logistic_Regression.ipynb," you'll find detailed explanations and coding examples demonstrating the utility of logistic regression algorithms.

# Introduction to Logistic Regression

![Logistic Regression](https://images.spiceworks.com/wp-content/uploads/2022/04/11040522/46-4.png)

Image from Vijay Kanade

Logistic regression is a popular statistical method used for binary classification tasks. Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability that an instance belongs to a particular class. It is widely used in various fields, including healthcare, finance, and marketing.

## Fundamentals of Logistic Regression

Logistic regression models the relationship between the independent variables and the probability of a binary outcome using the logistic function, also known as the sigmoid function. The output of the logistic function is bounded between 0 and 1, representing the probability of the positive class.

## Applications of Logistic Regression

Logistic regression finds applications in a wide range of domains, such as:
- Predicting whether a customer will churn or not in a telecommunications company.
- Determining whether a patient has a certain disease based on their medical history.
- Identifying fraudulent transactions in financial transactions.
- Classifying emails as spam or not spam based on their content.

## Advantages and Disadvantages of Logistic Regression

### Advantages:
- Simple and interpretable: Logistic regression provides interpretable coefficients that indicate the impact of each feature on the probability of the outcome.
- Efficient: Logistic regression is computationally efficient and can handle large datasets with ease.
- Well-understood: Logistic regression has been extensively studied and has a strong theoretical foundation.

### Disadvantages:
- Limited to binary classification: Logistic regression is suitable only for binary classification tasks and cannot be directly applied to multi-class problems without modification.
- Assumption of linearity: Logistic regression assumes a linear relationship between the independent variables and the log-odds of the outcome, which may not always hold true in practice.
- Vulnerable to overfitting: Logistic regression can overfit to noisy data or data with too many features, leading to poor generalization performance.

# Basics of Logistic Regression Algorithm

## Logistic Regression Model Representation

In logistic regression, the probability that an instance belongs to the positive class $y=1$ is modeled using the logistic function:

$$
P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}} \
$$

Where:
- $\mathbf{x}$ is the feature vector of the instance.
- $\mathbf{w}$ is the weight vector (coefficients) of the logistic regression model.

## Cost Function in Logistic Regression

The cost function in logistic regression is the log loss or cross-entropy loss, which measures the difference between the predicted probabilities and the actual class labels. The goal is to minimize the log loss to learn the optimal parameters of the logistic regression model.

## Gradient Descent in Logistic Regression

Gradient descent is the optimization algorithm used to minimize the cost function and learn the parameters of the logistic regression model. By iteratively updating the weights in the direction of the steepest descent of the cost function, gradient descent converges to the optimal solution.

# Model Evaluation and Parameter Tuning

## Choosing the Right Features

Feature selection is crucial in logistic regression to include only relevant features that contribute to the prediction while avoiding multicollinearity and overfitting. Techniques such as forward selection, backward elimination, and regularization can be used for feature selection.

## Evaluating Model Performance

Model performance in logistic regression is typically evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC curve. These metrics measure the classification performance of the logistic regression model and help assess its effectiveness in predicting the outcome.

## Tuning Hyperparameters in Logistic Regression

Hyperparameters in logistic regression include the regularization parameter $\lambda \$ for regularization techniques such as L1 (Lasso) and L2 (Ridge) regularization. Hyperparameter tuning involves selecting optimal values for these parameters to improve model performance and prevent overfitting.

