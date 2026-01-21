import matplotlib.pyplot as plt

def plot_ols(x_data, y_data, beta0, beta1):
    """
    Plot data points and a regression line.

    x_data, y_data: lists or arrays of your data points
    beta0: intercept
    beta1: slope
    """
    plt.figure(figsize=(8, 6))

    # Plot the data points
    plt.scatter(x_data, y_data, color='blue', label='Data')

    # Generate the regression line
    x_min, x_max = min(x_data), max(x_data)
    x_line = [x_min, x_max]
    y_line = [beta0 + beta1 * x for x in x_line]

    plt.plot(x_line, y_line, color='red', label=f'ŷ = {beta0} + {beta1}x')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage with my earlier data:
x = [3, -1, 7, -4, 10, 15]
y = [10.1, -2.5, 20.8, -20, 32, 47.2]
plot_ols(x, y, beta0=-2.368, beta1=3.3936)