import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np

# Sample data (as NumPy arrays)
data = np.array([
    [1, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [2, -2],
    [2, 2],
    [-2, 2],
    [-2, -2]
])

# Labels
labels = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

# Create a linear SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier
clf.fit(data, labels)

# Get the weight vector and bias from the trained model
w = clf.coef_[0]
b = clf.intercept_[0]

# Print the equation of the hyperplane
print(f"Hyperplane equation: {w[0]}*x1 + {w[1]}*x2 + {b} = 0")

# Extract hyperplane parameters
def get_hyperplane_params(w, b):
  """
  Calculates parameters for plotting the hyperplane.
  """
  # w = [w1, w2]
  # x1 = range of x values
  x1 = np.linspace(-3, 3, 100)
  
  # y = (-w[0] * x1 - b) / (w[1] + 1e-10)
  y = (-w[0] * x1 - b) / (w[1] + 1e-10)
  return x1, y

# Plot the data points
plt.scatter(data[:, 0], data[:, 1], c=labels)

# Plot the hyperplane
x1, y = get_hyperplane_params(w, b)
plt.plot(x1, y, color='k', linestyle='--')

# Set labels and title
plt.xlabel("x1")
plt.ylabel("x2")

# Show the plot
plt.grid(True)
plt.savefig("q2b.png")
plt.show()