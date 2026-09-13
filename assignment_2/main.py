import cv2
import numpy as np

image = cv2.imread("iris-1.jpg")
height, width, channels= image.shape

def padding(image, border_width):
    padded=cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/padded.png",padded)
    return padded

def crop(image, x_0, x_1, y_0, y_1):
    cropped=image[y_0:y_1, x_0:x_1]
    cv2.imwrite("solutions/cropped.png",cropped)
    return cropped

def resize(image, width, height):
    resized=cv2.resize(image, (width, height))
    cv2.imwrite("solutions/resized.png",resized)
    return resized

def copy(image, emptyPictureArray):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            emptyPictureArray[x][y]=image[x][y]
    cv2.imwrite("solutions/copy.png",emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/grayscale.png",gray)
    return gray

def hsv(image):
    hsv_image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/hsv.png",hsv_image)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(3):
                value=int(image[x][y][z])+hue
                if value>255:
                    value=255
                if value <0:
                    value=0
                emptyPictureArray[x][y][z]=value
    cv2.imwrite("solutions/hue_shifted.png",emptyPictureArray)

def smoothing(image):
    blurred=cv2.GaussianBlur(image, (15,15), cv2.BORDER_DEFAULT)
    cv2.imwrite("solutions/smoothed.png",blurred)
    return blurred

def rotation(image, rotation_angle):
    if rotation_angle==90:
        rotated=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle==180:
        rotated=cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite("solutions/rotated_" +str(rotation_angle)+ ".png",rotated)


padding(image, 100)
crop(image, 200, width -130, 200, height -130)
resize(image, 200, 200)

emptyPictureArray=np.zeros((height, width, 3), np.uint8)
copy(image, emptyPictureArray)
hue_shifted(image, emptyPictureArray, 50)
grayscale(image)
hsv(image)
smoothing(image)
rotation(image, 90)
rotation(image, 180)