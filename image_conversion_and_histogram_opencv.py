import cv2
import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------------------opencv code conversion --------------------------------------------------------
def rgb_to_hsv_opencv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def hsv_to_rgb_opencv(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def rgb_to_gray_opencv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb_to_ycbcr_opencv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb_opencv(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


def rgb_to_binary_opencv(img):
    gray = rgb_to_gray_opencv(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def hsv_to_binary_opencv(img):
    rgb = hsv_to_rgb_opencv(img)
    return rgb_to_binary_opencv(rgb)


def ycbcr_to_binary_opencv(img):
    rgb = ycbcr_to_rgb_opencv(img)
    return rgb_to_binary_opencv(rgb)


def gray_to_rgb_opencv(img_gray):
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


def show_img(img, imgresult):
    cv2.imshow('Before', img)
    cv2.imshow('After', imgresult)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------------------------------opencv code histogramme-------------------------------------------------------
def histogram_method_opencv2(path):

    img = cv2.imread(path)
    b, g, r = cv2.split(img)

    r_hist, _ = np.histogram(r, bins=256, range=(0, 256))
    g_hist, _ = np.histogram(g, bins=256, range=(0, 256))
    b_hist, _ = np.histogram(b, bins=256, range=(0, 256))

    return r_hist, g_hist, b_hist


def histogram_method_opencv2_ng(path):
    img = cv2.imread(path)
    ng = cv2.split(img)
    ng_hist, _ = np.histogram(ng, bins=256, range=(0, 256))
    return ng_hist

def histogramme_cumule_opencv(path):
    img = cv2.imread(path)
    b, g, r = cv2.split(img)
    r_hist, _ = np.histogram(r, bins=256, range=(0, 256))
    g_hist, _ = np.histogram(g, bins=256, range=(0, 256))
    b_hist, _ = np.histogram(b, bins=256, range=(0, 256))

    r_cum = np.cumsum(r_hist)
    g_cum = np.cumsum(g_hist)
    b_cum = np.cumsum(b_hist)

    return r_cum, g_cum, b_cum


def histogramme_cumule_opencv_ng(path):
    img = cv2.imread(path)
    ng = cv2.split(img)
    ng_hist, _ = np.histogram(ng, bins=256, range=(0, 256))
    return np.cumsum(ng_hist)