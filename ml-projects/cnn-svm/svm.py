# Source: https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
# Source: https://www.kaggle.com/code/jnikhilsai/digit-classification-using-svm-on-mnist-dataset

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import permutation_test_score

def train(dataset, gamma):
    # Load dataset
    digits = datasets.load_digits()

    # Prepare the data
    n_samples = len(digits.images)
    if dataset == "low_res_mnist":
        data = [resize(image, (7, 7)) for image in digits.images]
        data = [image.reshape(-1) for image in data]
    else:
        data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=gamma)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)

    # Learn the digits on the train subset
    classifier.fit(X_train, y_train)

    # Predict the value of the digit on the train and test subsets
    predicted_train = classifier.predict(X_train)
    predicted_test = classifier.predict(X_test)

    # Calculate training accuracy
    train_accuracy = accuracy_score(y_train, predicted_train)

    # Use permutation test to get test accuracy scores
    score, permutation_scores, pvalue = permutation_test_score(
        classifier, data, digits.target, cv=5, n_permutations=10, n_jobs=1)

    # Calculate the 95% confidence interval
    confidence_interval = (np.percentile(permutation_scores, 2.5), np.percentile(permutation_scores, 97.5))

    # Return training accuracy, permutation test score, and confidence interval
    return train_accuracy, score, confidence_interval

# Plot the SVM graphs
def plot_svm():
    gamma_values = np.logspace(-6, -1, 6)
    mnist_train_accuracies = []
    mnist_test_accuracies = []
    low_res_train_accuracies = []
    low_res_test_accuracies = []
    mnist_train_errors = []
    mnist_test_errors = []
    low_res_train_errors = []
    low_res_test_errors = []

    for gamma in gamma_values:
        mnist_train_acc, mnist_cv_acc, mnist_conf_interval = train("mnist", gamma)
        low_res_train_acc, low_res_cv_acc, low_res_conf_interval = train("low_res_mnist", gamma)
        mnist_train_accuracies.append(mnist_train_acc)
        mnist_test_accuracies.append(mnist_cv_acc)
        low_res_train_accuracies.append(low_res_train_acc)
        low_res_test_accuracies.append(low_res_cv_acc)
        mnist_train_errors.append(mnist_conf_interval[0])
        mnist_test_errors.append(mnist_conf_interval[1])
        low_res_train_errors.append(low_res_conf_interval[0])
        low_res_test_errors.append(low_res_conf_interval[1])
        
        print(f"Gamma: {gamma}")
        print(f"MNIST Test Accuracy: {mnist_cv_acc}")
        print(f"Low-res MNIST Test Accuracy: {low_res_cv_acc}")
        print()

    # Plotting for MNIST
    plt.figure(figsize=(10, 6))
    plt.errorbar(gamma_values, mnist_train_accuracies, yerr=mnist_train_errors, label='MNIST Train')
    plt.errorbar(gamma_values, mnist_test_accuracies, yerr=mnist_test_errors, label='MNIST Cross-validation')
    plt.xscale('log')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.title('Original MNIST Training vs Test Accuracy')
    plt.legend()
    plt.savefig('svm_mnist.png')
    plt.show()

    # Plotting for Low-res MNIST
    plt.figure(figsize=(10, 6))
    plt.errorbar(gamma_values, low_res_train_accuracies, yerr=low_res_train_errors, label='Low-res MNIST Train')
    plt.errorbar(gamma_values, low_res_test_accuracies, yerr=low_res_test_errors, label='Low-res MNIST Test')
    plt.xscale('log')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.title('Low Res MNIST Training vs Test Accuracy')
    plt.legend()
    plt.savefig('svm_low_res.png')
    plt.show()

if __name__ == "__main__":
    plot_svm()