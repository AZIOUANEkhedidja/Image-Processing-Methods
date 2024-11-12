import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from image_conversion_and_histogram_opencv import *
# ---------------------------------------------------code conversion-----------------------------------------------------------
def rgb_to_hsv(img):
    h = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    s = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    v = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    for i in range(len(img)):
        for j in range(len(img[i])):
            r = img[i][j][0] / 255.0
            g = img[i][j][1] / 255.0
            b = img[i][j][2] / 255.0
            
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            
            # Calculate Hue
            if delta == 0:
                h[i][j] = 0
            elif cmax == r:
                h[i][j] = 60 * (((g - b) / delta) % 6)
            elif cmax == g:
                h[i][j] = 60 * (((b - r) / delta) + 2)
            elif cmax == b:
                h[i][j] = 60 * (((r - g) / delta) + 4)
            
            # Calculate Saturation
            if cmax != 0:
                s[i][j] = delta / cmax
            else:
                s[i][j] = 0
            v[i][j] = cmax
    h = (h / 2)  
    s = (s * 255)  
    v = (v * 255)  
    
    return np.dstack((h, s, v)).astype(np.uint8) 


def hsv_to_rgb(hsv_image):
    height, width, _ = hsv_image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            h, s, v = hsv_image[i, j] / 255.0 
            a = 1.0
            if s:
                if h == 1.0: h = 0.0
                ii = int(h*6.0); f = h*6.0 - ii
                
                w = v * (1.0 - s)
                q = v * (1.0 - s * f)
                t = v * (1.0 - s * (1.0 - f))
                
                if ii==0: r, g, b, _ = (v, t, w, a)
                if ii==1: r, g, b, _ = (q, v, w, a)
                if ii==2: r, g, b, _ = (w, v, t, a)
                if ii==3: r, g, b, _ = (w, q, v, a)
                if ii==4: r, g, b, _ = (t, w, v, a)
                if ii==5: r, g, b, _ = (v, w, q, a)
            else: r, g, b, _ = (v, v, v, a)
            rgb_image[i, j] = [int(r * 255), int(g * 255), int(b * 255)]  

    return rgb_image.astype(np.uint8)


def rgb_to_gray(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    gray = np.zeros_like(r)
    for i in range(len(img)):
        for j in range(len(img[i])):
            gray[i][j] = 0.2989*r[i][j] + 0.5870*g[i][j] + 0.1140*b[i][j]
    return gray


def gray_to_rgb(img_gray):
    rgb = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            rgb[i][j] = [img_gray[i][j], img_gray[i][j], img_gray[i][j]]
    
    return rgb


def gray_to_binary(img_gray):
    seuil = 127
    binary = np.zeros_like(img_gray)
    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            if img_gray[i][j] > seuil:
                binary[i][j] = 255
            else:
                binary[i][j] = 0
                
    return binary


def binary_to_rgb(img_binary):
    rgb = np.zeros((img_binary.shape[0], img_binary.shape[1], 3), dtype=np.uint8)
    for i in range(len(img_binary)):
        for j in range(len(img_binary[i])):
            rgb[i][j] = [img_binary[i][j], img_binary[i][j], img_binary[i][j]]
    
    return rgb


def rgb_to_ycbcr(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    y = np.zeros_like(r)
    cb = np.zeros_like(r)
    cr = np.zeros_like(r)
    for i in range(len(img)):
        for j in range(len(img[i])):
            y[i][j] = 0.299*r[i][j] + 0.587*g[i][j] + 0.114*b[i][j]
            cb[i][j] = 128 - 0.168736*r[i][j] - 0.331264*g[i][j] + 0.5*b[i][j]
            cr[i][j] = 128 + 0.5*r[i][j] - 0.418688*g[i][j] - 0.081312*b[i][j]

    return np.dstack((y,cb,cr))


def ycbcr_to_rgb(img):
    y = img[:,:,0]
    cb = img[:,:,1]
    cr = img[:,:,2]
    r = np.zeros_like(y)
    g = np.zeros_like(y)
    b = np.zeros_like(y)
    for i in range(len(img)):
        for j in range(len(img[i])):
            r[i][j] = y[i][j] + 1.402*(cr[i][j]-128)
            g[i][j] = y[i][j] - 0.344136*(cb[i][j]-128) - 0.714136*(cr[i][j]-128)
            b[i][j] = y[i][j] + 1.772*(cb[i][j]-128)
            
    return np.dstack((r,g,b))

# ------------------------------------------------------code histogramme--------------------------------------------------------
def histogram_method(path):
    # if isinstance(path, str):
    image = Image.open(path)
    hist_r = [0] * 256
    hist_g = [0] * 256
    hist_b = [0] * 256
    pixels = list(image.getdata())
    for pixel in pixels:
        r, g, b = pixel
        hist_r[r] += 1
        hist_g[g] += 1
        hist_b[b] += 1

    return hist_r, hist_g, hist_b


def histogram_method_ng(path_or_img):
    if isinstance(path_or_img, str):
        image = Image.open(path_or_img)
    elif isinstance(path_or_img, np.ndarray):
        image = Image.fromarray(path_or_img)
    elif isinstance(path_or_img, Image.Image):
        image = path_or_img
    hist_r = [0] * 256
    pixels = list(image.getdata())
    for pixel in pixels:
        r= pixel
        # print(hist_r)
        hist_r[r] += 1
    return hist_r


def histogramme_cumule_ng(path_or_img):
    tableaux_ng  = histogram_method_ng(path_or_img)
    tableaux_cumule_ng = [0] * 256
    tableaux_cumule_ng[0] = tableaux_ng[0]
    for i in range(1, 256):
        tableaux_cumule_ng[i] = tableaux_cumule_ng[i-1] + tableaux_ng[i]
    return tableaux_cumule_ng


def histogramme_cumule(path):
    tableaux_r, tableaux_g, tableaux_b = histogram_method(path)
    
    tableaux_cumule_r = [0] * 256
    tableaux_cumule_g = [0] * 256
    tableaux_cumule_b = [0] * 256
    
    tableaux_cumule_r[0] = tableaux_r[0]
    tableaux_cumule_g[0] = tableaux_g[0]
    tableaux_cumule_b[0] = tableaux_b[0]
    
    for i in range(1, 256):
        tableaux_cumule_r[i] = tableaux_cumule_r[i-1] + tableaux_r[i]
        tableaux_cumule_g[i] = tableaux_cumule_g[i-1] + tableaux_g[i]
        tableaux_cumule_b[i] = tableaux_cumule_b[i-1] + tableaux_b[i]

    return tableaux_cumule_r, tableaux_cumule_g, tableaux_cumule_b


def etirement_histogramme(path):
    img = cv2.imread(path)
    min_val = np.min(img)
    max_val = np.max(img)
    image_etiree = (img - min_val) * (255 / (max_val - min_val))
    return image_etiree


def egalisation_histogramme(path):
    img = cv2.imread(path)
    img_source = rgb_to_gray(img)
    
    hist_cumule = histogramme_cumule_ng(img_source)
    
    total_pixels = img_source.shape[0] * img_source.shape[1]
    img_destination = np.zeros_like(img_source)

    for i in range(256):
        hist_cumule[i] = (hist_cumule[i] * 255) // total_pixels
    
    for i in range(img_source.shape[0]):
        for j in range(img_source.shape[1]):
            img_destination[i][j] = hist_cumule[img_source[i][j]]
    return img_destination


def distance_histogramme(path1,path2,distance_name):
    hist1 = histogram_method_ng(path1)
    hist2 = histogram_method_ng(path2)
    if distance_name == 'Intersection':
        return distance_Intersection(hist1, hist2)
    elif distance_name == 'Correlation':
        return distance_Correlation(hist1, hist2)
    elif distance_name == 'Chi-Square':
        return distance_Chi_deux(hist1, hist2)
    else:
        return "Distance not found"


def distance_Intersection(hist1, hist2):
    hist1 = hist1/np.sum(hist1)
    hist2 = hist2/np.sum(hist2)
    return np.sum(np.minimum(hist1, hist2))


def distance_Correlation(hist1, hist2):
    hist1 = hist1 - np.mean(hist1)
    hist2 = hist2 - np.mean(hist2)
    return np.sum(hist1 * hist2) / (np.sqrt(np.sum(hist1**2)) * np.sqrt(np.sum(hist2**2)))


def distance_Chi_deux(hist1, hist2):
    hist1 = np.array(hist1)
    hist2 = np.array(hist2)
    return np.sum((hist1 - hist2) ** 2 / (hist2+ 1e-10) )


def distance_histogramme_joint(path1,path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    img1 = rgb_to_gray(img1)
    img2 = rgb_to_gray(img2)
    p = np.zeros((256,256))
    for i in  range(len(img1)):
        for j in range(len(img2)):
            p[img1[i][j]][img2[i][j]] += 1
    return p

