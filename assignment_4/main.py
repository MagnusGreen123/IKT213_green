import cv2
import numpy as np


def harris(reference_image):
    corner_pic = reference_image.copy()
    gray_img = cv2.cvtColor(corner_pic, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    corner_map = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    corner_map = cv2.dilate(corner_map, None)

    corner_pic[corner_map > 0.01 * corner_map.max()] = [0, 0, 255]

    cv2.imwrite("solutions/harris.png", corner_pic)


def align(image_to_align, reference_image, max_features, good_match_precent):
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_align, desc_align = sift.detectAndCompute(gray_align, None)
    kp_ref, desc_ref = sift.detectAndCompute(gray_ref, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    all_matches = flann.knnMatch(desc_align, desc_ref, k=2)

    treff = []
    for m, n in all_matches:
        if m.distance < good_match_precent * n.distance:
            treff.append(m)

    if len(treff) > max_features:
        pts_align = np.float32([kp_align[m.queryIdx].pt for m in treff]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in treff]).reshape(-1, 1, 2)

        h_matrix, mask = cv2.findHomography(pts_align, pts_ref, cv2.RANSAC, 5.0)
        inliers = mask.ravel().tolist()

        height, width, channels = reference_image.shape
        aligned_pic = cv2.warpPerspective(image_to_align, h_matrix, (width, height))
        cv2.imwrite("solutions/aligned.png", aligned_pic)
    else:
        print("Not enough matches found - %d/%d" % (len(treff), max_features))
        inliers = None

    draw_params = dict(matchColor=(0, 0, 255),
                       singlePointColor=None,
                       flags=2)

    best_ones = sorted(treff, key=lambda x: x.distance)[:10]
    match_pic = cv2.drawMatches(image_to_align, kp_align, reference_image, kp_ref, best_ones, None, **draw_params)
    cv2.imwrite("solutions/matches.png", match_pic)


reference_image = cv2.imread("reference_img.png")
image_to_align = cv2.imread("align_this.jpg")

harris(reference_image)

align(image_to_align, reference_image, 10, 0.7)