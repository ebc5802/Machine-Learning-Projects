import matplotlib.pyplot as plt

# Data for the first set of points
x1_positive = [2, 3, 3]
x2_positive = [1, 2, 0]

# Data for the second set of points
x1_negative = [1, 2, 1]
x2_negative = [0, 0, 1]

# Plot the first set of points in blue
plt.scatter(x1_positive, x2_positive, color='green', label='Group 1')

# Plot the second set of points in red
plt.scatter(x2_negative, x2_negative, color='red', label='Group 2')

# Add labels, title, and legend
plt.xlabel('x1-axis')
plt.ylabel('x2-axis')
plt.title('Positive (Green) vs. Negative (Red)')
plt.legend()

# Show the plot
plt.show()