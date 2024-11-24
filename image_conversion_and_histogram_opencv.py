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


def histogram_method_opencv2_ng(path_or_img):
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img)
    elif isinstance(path_or_img, np.ndarray):
        img = path_or_img
    # img = cv2.imread(path_or_img)
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


def normalization_histogramme_opencv(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) 
    min_val = np.min(image)
    max_val = np.max(image)
    image_etiree = cv2.normalize(image, None, min_val, max_val, cv2.NORM_MINMAX)
    return image_etiree


def egalisation_histogramme_opencv(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)
    return image


def distance_histogramme_opencv(path_img1,path_img2,distance_name):
    hist1 = histogram_method_opencv2_ng(path_img1)
    hist2 = histogram_method_opencv2_ng(path_img2)
    # Normalize histograms
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)
    
    if distance_name == 'Intersection':
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    elif distance_name == 'Correlation':
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    elif distance_name == 'Chi-Square':
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    else:
        return "Invalid distance name"

def histogramme_joint_opencv(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    hist_2d, x_edges, y_edgs = np.histogram2d(img1.ravel(), img2.ravel(), bins=256, range=[[0, 256], [0, 256]])
    return hist_2d


def distance_histogramme_joint_opencv(path1, path2):
    hist = histogramme_joint_opencv(path1, path2)
    hist = hist / np.sum(hist)
    information_mutuelle = 0
    for i in range(256):
        for j in range(256):
            if hist[i][j] > 0:
                prob_i = np.sum(hist[i, :])  
                prob_j = np.sum(hist[:, j])  
                information_mutuelle += hist[i][j] * np.log(hist[i][j] / (prob_i * prob_j))

    return information_mutuelle

# -------------------------------------------------------opencv code Transformations géométriques-------------------------------------------------------

def translate_opencv(image, tx, ty):
    height, width = image.shape[:2]
    translation_matrix = np.array([
        [1, 0, tx],  
        [0, 1, ty]   
    ], dtype=np.float32)
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated_image


def rotation_ppv_opencv(path, angle, interpolation=cv2.INTER_LINEAR):
    img = cv2.imread(path)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_res = cv2.warpAffine(img, M, (cols, rows), flags=interpolation)
    return img_res


def echelle_opencv(path, scale_x, scale_y):
    img = cv2.imread(path)
    img_res = cv2.resize(img, None, fx=scale_x, fy=scale_y)
    return img_res


def cisaillement_opencv(path, shx, shy):
    img = cv2.imread(path)
    rows, cols = img.shape[:2]
    shear_matrix = np.array([
        [1, shx, 0],
        [shy, 1, 0]
    ], dtype=np.float32)
    img_res = cv2.warpAffine(img, shear_matrix, (cols, rows))
    return img_res
