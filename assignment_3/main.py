import cv2
import numpy as np


def sobel_edge_detection(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobelxy = cv2.normalize(sobelxy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite("soultions/sobel_edge_detection.png", sobelxy)
    return sobelxy


def canny_edge_detection(image, threshold_1, threshold_2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold_1, threshold2=threshold_2)

    cv2.imwrite("soultions/canny_edge_detection.png", edges)
    return edges


def template_match(image, template):
    img_rgb = image.copy()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("soultions/template_match.png", img_rgb)
    return img_rgb


def resize(image, scale_factor: int, up_or_down: str):
    rows, cols, _channels = map(int, image.shape)

    if up_or_down == "up":
        resized = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
    elif up_or_down == "down":
        resized = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
    else:
        print("up_or_down must be 'up' or 'down'")
        return image

    cv2.imwrite("soultions/resize_" + up_or_down + ".png", resized)
    return resized


lambo = cv2.imread("lambo.png")
shapes = cv2.imread("shapes-1.png")
shapes_template = cv2.imread("shapes_template.jpg")
sobel_edge_detection(lambo)
canny_edge_detection(lambo, 50, 50)
template_match(shapes, shapes_template)
resize(lambo, 2, "up")
resize(lambo, 2, "down")