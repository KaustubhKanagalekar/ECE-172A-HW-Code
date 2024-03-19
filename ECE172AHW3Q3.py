######
import numpy as np 
import math 
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

ksmooth = 1/159 * np.array([[2,4,5,4,2], [4,9,12,9,4], [5, 12, 15, 12, 5], [4,9,12,9,4] ,[2,4,5,4,2]])
kx = np.array([[-1,0,1], [-2, 0, 2], [-1, 0,1]])
ky = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

def mag_and_dir(Gx,Gy): 
    mag_G = np.sqrt(Gx**2 + Gy**2)
    dir_G = (np.arctan2(Gy, Gx))
    dir_G = (np.degrees(dir_G) + 360) % 360

    return mag_G, dir_G
#################################################
def canny_edge(mag, dir):
    x, y = mag.shape
    remove = np.zeros(mag.shape)
    for i in range(1, x-1):
        for j in range(1, y-1):
            angle = dir[i,j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (337.5 <= angle < 360):
                neighbour1 = mag[i, j+1]
                neighbour2 = mag[i, j-1]
            elif (22.5 <= angle < 67.5) or (202.5 <= angle < 247.5):
                neighbour1 = mag[i+1, j-1]
                neighbour2 = mag[i-1, j+1]
            elif (67.5 <= angle < 112.5) or (247.5 <= angle < 292.5):
                neighbour1 = mag[i+1, j]
                neighbour2 = mag[i-1, j]
            elif (112.5 <= angle < 157.5) or (292.5 <= angle < 337.5):
                neighbour1 = mag[i-1, j-1]
                neighbour2 = mag[i+1, j+1]

            if mag[i, j] >= neighbour1 and mag[i, j] >= neighbour2:
                remove[i, j] = mag[i, j]
            else:
                remove[i, j] = 0

    return remove 

image = plt.imread('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/lane.png')

gray_image = image.mean(axis=2)
#gray_image = plt.imshow(image, cmap= 'gray')
plt.imshow(gray_image, cmap = 'gray')
print(image.dtype)
plt.show()

test_img = convolve(gray_image, ksmooth)
Gx = convolve(test_img, kx)
Gy = convolve(test_img, ky)

mag, direction = mag_and_dir(Gx, Gy)
edges = canny_edge(mag, direction)
plt.imshow(edges, cmap = 'gray')
print(edges.dtype)
plt.show()

def threshold(edges, number):
    improve_edges = np.copy(edges)
    improve_edges[improve_edges < number] = 0 
    return improve_edges 

adjusted_edges = threshold(edges, 0.0000000001)
plt.imshow(adjusted_edges, cmap = 'gray')
print(adjusted_edges.dtype)
plt.show()


