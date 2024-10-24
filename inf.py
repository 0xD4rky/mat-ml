import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r'/home/darky/Documents/formula1/10000.dat', delimiter='\t')

image = np.zeros((256, 256), dtype=np.int32)  
for row in data:
    i, j = int(row[0]), int(row[1])  # Get pixel coordinates (i, j)
    class_value = int(row[5])  # Get the class value (6th column)
    image[i, j] = class_value  # Assign the class value to the corresponding pixel

plt.imshow(image, cmap='tab20')  
plt.colorbar()  
plt.title("Image showing different classes")
plt.axis('off')  

plt.savefig('class_image.png')

print("Image saved as 'class_image.png'.")
