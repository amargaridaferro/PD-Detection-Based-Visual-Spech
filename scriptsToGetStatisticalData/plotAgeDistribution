import matplotlib.pyplot as plt
import seaborn as sns

# Sample ages for two sets
set1_ages = [60, 39, 59, 53, 48, 37, 41, 33, 49, 31, 52, 39, 55, 35, 28, 55, 43, 65, 61, 63, 41, 39, 65, 31, 33, 43, 35, 29, 38, 31, 35, 50, 28, 35, 48, 35, 25, 30, 33, 60, 37, 45, 57, 35, 51, 55]

set2_ages = [53, 45, 45, 39, 43, 31, 25, 59, 37, 49, 33, 38, 38, 43, 48, 43, 31, 25, 50, 59, 63, 33, 35, 37, 51, 33, 45, 57, 63, 70, 59, 47, 31, 31, 39, 27, 30, 35, 53, 37, 33, 37, 39, 39, 31, 59, 31, 29]

# Create KDE plots
sns.kdeplot(set1_ages, label='PD')
sns.kdeplot(set2_ages, label='HC')

# Set the x-axis label
plt.xlabel('Age')

# Set a title for the plot

# Show the legend
plt.legend()

# Show the plot
plt.show()
