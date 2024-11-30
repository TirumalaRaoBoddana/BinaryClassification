import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
# Generate synthetic data for binary classification
X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=1,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           n_classes=2,
                           random_state=41, class_sep=0.8)
def get_dataset():
    return X,y
# Perceptron algorithm implementation
def perceptron(X, y, epochs, learning_rate):
    X = np.insert(X, 0, 1, axis=1)  # Insert bias term (1) into the features
    coefficients = np.ones(X.shape[1])  # Initialize coefficients
    weights_history = []
    for i in range(epochs):
        random_index = np.random.randint(0, X.shape[0])
        y_hat = np.dot(X[random_index], coefficients)
        predicted_value = 1 if y_hat >= 0 else 0  # Step function
        coefficients = coefficients + learning_rate * (y[random_index] - predicted_value) * X[random_index]
        weights_history.append(coefficients.copy())  # Save a copy of the coefficients at each step
    return weights_history
# Main function to set up plot and animation
def main():
    #change the epochs and learning rate 
    epochs = 1000
    learning_rate = 0.01
    weights_history = perceptron(X, y, epochs, learning_rate)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter plot for the data points
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    # Line for the decision boundary (initially empty)
    line, = ax.plot([], [], color='green')
    # Set labels and title
    ax.set_title("Perceptron Decision Boundary Movement")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    # Text elements for dynamic update of slope, intercept, and epoch
    slope_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='purple')
    intercept_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, color='purple')
    epoch_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=12, color='purple')
    return fig, ax, line, weights_history, slope_text, intercept_text, epoch_text
# Initialize the plot
def init():
    fig, ax, line, weights_history, slope_text, intercept_text, epoch_text = main()
    line.set_data([], [])
    slope_text.set_text('')
    intercept_text.set_text('')
    epoch_text.set_text('')
    return line, slope_text, intercept_text, epoch_text
# Function to update the decision boundary at each frame (epoch)
def update(epoch, line, weights_history, slope_text, intercept_text, epoch_text):
    weights = weights_history[epoch]
    # Slope and intercept of the decision boundary
    slope = -weights[1] / weights[2]
    intercept = -weights[0] / weights[2]
    # Generate x values and calculate corresponding y values
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = slope * x_values + intercept
    # Update the line
    line.set_data(x_values, y_values)
    # Update the slope, intercept, and epoch text
    slope_text.set_text(f'Slope: {slope:.4f}')
    intercept_text.set_text(f'Intercept: {intercept:.4f}')
    epoch_text.set_text(f'Epoch: {epoch + 1}')
    return line, slope_text, intercept_text, epoch_text
# Animation function
def animate_perceptron():
    fig, ax, line, weights_history, slope_text, intercept_text, epoch_text = main()
    anim = FuncAnimation(fig, update, frames=len(weights_history), 
                         fargs=(line, weights_history, slope_text, intercept_text, epoch_text), 
                         init_func=init, interval=50, blit=True)
    #Save the animation as a GIF
    anim.save("perceptron_animation.gif", writer='pillow')
    print("successfully saved the video")
animate_perceptron()
