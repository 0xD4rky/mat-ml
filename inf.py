import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data from the .dat file, adjusting the delimiter for tabs
data = np.loadtxt(r'/home/darky/Documents/formula1/0.dat', delimiter='\t')

# Step 2: Create an empty array for the image
image = np.zeros((256, 256), dtype=np.int32)  # To hold the class values

# Step 3: Populate the image array with class values
for row in data:
    i, j = int(row[0]), int(row[1])  # Get pixel coordinates (i, j)
    class_value = int(row[5])  # Get the class value (6th column)
    image[i, j] = class_value  # Assign the class value to the corresponding pixel

# Step 4: Plot the image with a color map to distinguish different classes
plt.imshow(image, cmap='tab20')  # Using 'tab20' colormap which has 20 distinct colors
plt.colorbar()  # Show color bar to represent classes
plt.title("Image showing different classes")
plt.axis('off')  # Hide the axis
plt.show()
