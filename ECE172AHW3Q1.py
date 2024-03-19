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


def computeNormRGBHistogram(image):
    img = cv.imread(image)
    plt.imshow(img)
    plt.show()
    b,g,r = cv.split(img)
    
    bin_width = 256/32

    red_hist = np.zeros(32, dtype = int)
    green_hist = np.zeros(32, dtype = int)
    blue_hist = np.zeros(32, dtype = int)

    for pixel in b.flatten():
        pixel_grp = int(pixel/bin_width)
        blue_hist[pixel_grp] +=1 
    for pixel in g.flatten():
        pixel_grp = int(pixel/bin_width)
        green_hist[pixel_grp] +=1 
    for pixel in r.flatten():
        pixel_grp = int(pixel/bin_width)
        red_hist[pixel_grp] +=1 

    red_hist = red_hist/(np.sum(red_hist))
    blue_hist = blue_hist/(np.sum(blue_hist))
    green_hist = green_hist/(np.sum(green_hist))

    
    total_hist = np.concatenate((red_hist, green_hist, blue_hist))
    plt.bar(np.arange(96), total_hist, color = ['red']*32 + ['green']*32 + ['blue']*32)
    plt.show()

    return total_hist

computeNormGrayHistogram('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/forest.jpg')
computeNormRGBHistogram('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/forest.jpg')

def image_flipped(image_path):
    image = cv.imread(image_path)
    flipped_image = np.fliplr(image)
    return flipped_image
image_path = '/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/forest.jpg'
flipped_image_path = '/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/flipped_forest.jpg'

flipped_image = image_flipped(image_path)
cv.imwrite(flipped_image_path, flipped_image) 
computeNormGrayHistogram(flipped_image_path)

forest_image = '/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/forest.jpg'
change_forest_image = cv.imread(forest_image)
change_forest_image[:,:,2] = np.clip(change_forest_image[:, :, 2] * 2, 0, 255)

temp_image_path = '/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/temp_forest.jpg'
cv.imwrite(temp_image_path, change_forest_image)
computeNormRGBHistogram(temp_image_path)

def adaptiveHist2(image, winSize):
    img = cv.imread(image, 0)
    height, width = img.shape
    output = np.zeros((height, width))

    new_img = np.array((height,width))
    extrapad = np.pad(img, (winSize//2), mode = 'reflect')

    for y in range(width):
        for x in range(height):
            rank = 0 
            context_region = extrapad[x:(x+winSize), y:(y+winSize)]
            for i in range(winSize):
                for j in range(winSize):
                    if context_region[i, j] < img[x, y]:
                        rank = rank + 1
            output[x, y] = rank * 255 / (winSize ** 2)
            new_img = output 

    plt.imsave("Q1output.png", new_img, cmap= 'gray')   
    plt.imshow(new_img, cmap= 'gray')     
    return new_img



#adaptiveHist2('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/beach.png', 33)
#adaptiveHist2('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/beach.png', 65)    
#adaptiveHist2('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/beach.png', 129)   
computeNormGrayHistogram('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/beach.png')