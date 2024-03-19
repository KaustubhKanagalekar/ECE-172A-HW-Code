import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

def computeNormGrayHistogram(image):
    gray_image = cv.imread(image,0)
    plt.imshow(gray_image)
    plt.show()

    bin_width = 256/32 
    histogram = np.zeros(32, dtype = int) 
    
    for pixel in gray_image.flatten():
        pixel_grp = int(pixel/bin_width)
        histogram[pixel_grp] +=1
    
    histogram = histogram/np.sum(histogram)
    plt.bar(np.arange(32), histogram)
    plt.show()
    
    return histogram 

computeNormGrayHistogram('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise1.jpg')
computeNormGrayHistogram('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise2.jpg')

#######################################################

def mean_filter(image, size):
    image = cv.imread(image, 0)
    extrapad = np.pad(image, (size//2), mode = 'constant')
    filter_image = np.zeros_like(image)
    height, width = image.shape 

    for i in range(height):
        for j in range(width):
            neighbourhood = extrapad[i:i+size, j:j+size]
            filter_image[i,j] = np.sum(neighbourhood)/(size**2)
    plt.show()

    bin_width = 256/32 
    histogram = np.zeros(32, dtype = int) 
    
    for pixel in image.flatten():
        pixel_grp = int(pixel/bin_width)
        histogram[pixel_grp] +=1
    
    histogram = histogram/np.sum(histogram)
    plt.bar(np.arange(32), histogram)
    plt.show()
    plt.imsave("Q3O1.png", filter_image, cmap= 'gray')  
    
    return histogram, filter_image
    


def median_filter(image, size):
    image = cv.imread(image,0)
    extrapad = np.pad(image, (size//2), mode = 'constant')
    filter_image = np.zeros_like(image)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            neighbourhood = extrapad[i:i+size , j:j+size]
            filter_image[i,j] = np.median(neighbourhood)
    plt.show()
    
    bin_width = 256/32 
    histogram = np.zeros(32, dtype = int) 
    
    for pixel in image.flatten():
        pixel_grp = int(pixel/bin_width)
        histogram[pixel_grp] +=1
    
    histogram = histogram/np.sum(histogram)
    plt.bar(np.arange(32), histogram)
    plt.show()
    plt.imsave("Q3O2.png", filter_image, cmap= 'gray')  
    return histogram, filter_image

#mean_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise1.jpg', 5)
#mean_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise1.jpg', 81)
#median_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise1.jpg', 5)
#median_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise1.jpg', 81)

#mean_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise2.jpg', 5)
#mean_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise2.jpg', 81)
#median_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise2.jpg', 5)
#median_filter('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural_noise2.jpg', 81)

#####################################

img = cv.imread('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural.jpg', cv.IMREAD_GRAYSCALE)
template = cv.imread('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/template.jpg', cv.IMREAD_GRAYSCALE)

result = cv.matchTemplate(img, template, cv.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

h, w = template.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img, top_left, bottom_right, 255, 2)

plt.imshow(img, cmap='gray')
plt.show()

######################################
img = cv.imread('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/mural.jpg', cv.IMREAD_GRAYSCALE)
template = cv.imread('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/template.jpg', cv.IMREAD_GRAYSCALE)

result = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

h, w = template.shape

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(img, top_left, bottom_right, 255, 2)
plt.imshow(img, cmap='gray')
plt.show()
