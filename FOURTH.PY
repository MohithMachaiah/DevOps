import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
mtcars = pd.read_csv('mtcars.csv')  # Replace 'path_to_your_mtcars.csv' with the actual path to your mtcars.csv file

# Plotting the histogram
plt.hist(mtcars['mpg'], bins=17, color='green', edgecolor='black')

# Adding labels and title
plt.xlabel('Miles per gallon (mpg)')
plt.ylabel('Frequency')
plt.title('Histogram of Miles per gallon (mpg)')

# Displaying the plot
plt.show()