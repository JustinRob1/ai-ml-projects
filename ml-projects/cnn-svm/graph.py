from cnn import run_test
from svm import plot_svm
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Plot the CNN graphs
def plot_cnn(histories, filename):
    train_acc = np.array([history.history['accuracy'] for history in histories])
    train_acc_mean = np.mean(train_acc, axis=0)
    train_acc_std = np.std(train_acc, axis=0)

    test_acc = np.array([history.history['val_accuracy'] for history in histories])
    test_acc_mean = np.mean(test_acc, axis=0)
    test_acc_std = np.std(test_acc, axis=0)

    epochs = np.arange(1, len(train_acc_mean) + 1)

    pyplot.figure(figsize=(10, 6))
    pyplot.plot(epochs, train_acc_mean, color='blue', label='Mean Training Accuracy')
    pyplot.fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, color='lightblue', alpha=0.5)

    pyplot.plot(epochs, test_acc_mean, color='orange', label='Test Accuracy')
    pyplot.fill_between(epochs, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, color='bisque', alpha=0.5)

    if filename == "low_res_mnist":
        pyplot.title('Low Res MNIST Traning vs Test Accuracy')
    else:
        pyplot.title('Fashion MNIST Traning vs Test Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend(loc="lower right")
    pyplot.savefig(filename + '_plot.png')
    pyplot.show()

def main():
    histories = run_test("fashion_mnist") 
    plot_cnn(histories, "fashion_mnist") 
    histories = run_test("low_res_mnist")
    plot_cnn(histories, "low_res_mnist")  
    plot_svm()
    
if __name__ == "__main__":
    main()