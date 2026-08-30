import numpy as np
import cv2

image = cv2.imread('iris-1.jpg',1)


def print_image_information(image):
    print("Height:", image.shape[0])
    print("Width:", image.shape[1])
    print("Channels:", image.shape[2])
    print("Size:", image.size)
    print("Data type:", image.dtype)

print_image_information(image)


cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps= int(cam.get(cv2.CAP_PROP_FPS))

cam.release()

with open("solutions/camera_outputs.txt", "w") as f:
    f.write("fps: " + str(cam_fps) + "\n")
    f.write("frame_height: " + str(frame_height) + "\n")
    f.write("frame_width: " + str(frame_width))
