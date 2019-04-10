""" extract features
"""
from time import time
import cv2
import numpy as np
from datetime import datetime
from glob import glob
from sklearn import svm
from pathlib import Path
from numba import jit
from time import time
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from skimage.feature import local_binary_pattern
from skimage.transform import rescale

from storage import save, load

def cropFace(image):
    """ crop face from a given image
    """
    faceClassifier = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_alt.xml')
    faces = faceClassifier.detectMultiScale(image)
    #get first face coordinates
    xFace, yFace, wFace, hFace = faces[0]
    croppedFace = image[yFace:yFace+hFace, xFace:xFace+wFace]
    return croppedFace

def kirsch(image):
    """ compute the 8 filters of kirsch
    """
    return {
        "N" : cv2.filter2D(image, -1, np.array([[-3, -3, -3],
                                                [-3,  0, -3],
                                                [ 5,  5,  5]
                                               ])),
        "NE" : cv2.filter2D(image, -1, np.array([[-3, -3, -3],
                                                 [ 5,  0, -3],
                                                 [ 5,  5, -3]
                                                ])),
        "E" : cv2.filter2D(image, -1, np.array([[5, -3, -3],
                                                [5,  0, -3],
                                                [5, -3, -3]
                                               ])),
        "SE" : cv2.filter2D(image, -1, np.array([[ 5,  5, -3],
                                                 [ 5,  0, -3],
                                                 [-3, -3, -3]
                                                ])),
        "S" : cv2.filter2D(image, -1, np.array([[ 5,  5,  5],
                                                [-3,  0, -3],
                                                [-3, -3, -3]
                                               ])),
        "SW" : cv2.filter2D(image, -1, np.array([[-3,  5,  5],
                                                 [-3,  0,  5],
                                                 [-3, -3, -3]
                                                ])),
        "W" : cv2.filter2D(image, -1, np.array([[-3, -3, 5],
                                                [-3,  0, 5],
                                                [-3, -3, 5]
                                               ])),
        "NW" : cv2.filter2D(image, -1, np.array([[-3, -3, -3],
                                                 [-3,  0,  5],
                                                 [-3,  5,  5]
                                                ]))
    }

@jit(nopython=True)
def dLBPPixel(image, angle):
    """ compute dLBP pixel value
    """
    if angle==0:
        elements = np.array([image[4, 8], image[4, 7], image[4, 6], image[4, 5],
                             image[4, 4], image[4, 3], image[4, 2], image[4, 1],
                             image[4, 0]], dtype=np.uint8)
    elif angle==45:
        elements = np.array([image[8, 0], image[7, 1], image[6, 2], image[5, 3],
                             image[4, 4], image[3, 5], image[2, 6], image[1, 7],
                             image[0, 8]], dtype=np.uint8)
    elif angle==90:
        elements = np.array([image[8, 4], image[7, 4], image[6, 4], image[5, 4],
                             image[4, 4], image[3, 4], image[2, 4], image[1, 4],
                             image[0, 4]], dtype=np.uint8)
    elif angle==135:
        elements = np.array([image[8, 8], image[7, 7], image[6, 6], image[5, 5],
                             image[4, 4], image[3, 3], image[2, 2], image[1, 1],
                             image[0, 0]], dtype=np.uint8)
    return ((elements[4] > elements[8])<<0) + \
           ((elements[4] > elements[7])<<1) + \
           ((elements[4] > elements[6])<<2) + \
           ((elements[4] > elements[5])<<3) + \
           ((elements[4] > elements[3])<<4) + \
           ((elements[4] > elements[2])<<5) + \
           ((elements[4] > elements[1])<<6) + \
           ((elements[4] > elements[0])<<7)

@jit(nopython=True)
def dLBP(image, angle=0):
    """ compute directional LBP
    """
    height = image.shape[0]
    width = image.shape[1]
    result = np.empty((height - 8, width - 8), dtype=np.uint8)
    for w in range(4, width - 4):
        for h in range(4, height - 4):
            subImg = image[h-4:h+5, w-4:w+5]
            result[h-4, w-4] = dLBPPixel(subImg, angle)
    return result

@jit(nopython=True)
def generatePattern(matrix):
    """ generate pattern of LBP (clockwise)
    """
    return matrix[0, 0] + (matrix[0, 1]<<1) + (matrix[0, 2]<<2) + (matrix[1, 2]<<3) + \
           (matrix[2, 2]<<4) + (matrix[2, 1]<<5) + (matrix[2, 0]<<6) + (matrix[1, 0]<<7)

def hvnLBPWithoutBorders(image):
    """ computer horizontal and vertical LBP ( hvnLBP )
    """
    height, width = image.shape
    newImage = np.empty([height -2, width -2], dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            mat = image[i-1:i+2, j-1:j+2].copy()
            mat[1, 1] = 0
            #vertical LBP
            vmat = mat == mat.max(axis=0).reshape(1, 3)
            #horizontal LBP
            hmat = mat == mat.max(axis=1).reshape(3, 1)
            
            hv = np.maximum(vmat, hmat)
            newImage[i-1, j-1] = generatePattern(hv)
    return newImage

@jit(nopython=True)
def LBPH(img, gridX, gridY):
    """ generate LBP histogram
    """
    cellWidth = img.shape[1] // gridX
    cellHeight = img.shape[0] // gridY
    
    feature = np.array((), dtype=np.uint8)
    for x in range(0, gridX*cellWidth, cellWidth):
        for y in range(0, gridY*cellHeight, cellHeight):
            # extract cell
            subImg = img[y:y+cellHeight, x:x+cellWidth]
            # construct histogram
            hist, _binEdge = np.histogram(subImg.ravel(), bins=256)
            # concatenate histograms 
            feature = np.concatenate((feature, hist.astype(np.uint8)))
    return feature

def extractFeature(image):
    """ extract features from image
    """
    # preprocessing
    image = cv2.equalizeHist(image)
    for _ in range(5):
        image = cv2.bilateralFilter(image, 9, 250, 250)
    # extract face
    image = cropFace(image)
    # rescale to 77x77 (without borders => 75x75)
    image = rescale(image, 77.0/image.shape[0], anti_aliasing=True, multichannel=False)
    # filteredImages = kirsch(image)
    feature = np.array([], dtype=np.uint8)
    # for _key, filteredImage in filteredImages.items():
    lbp = hvnLBPWithoutBorders(image)
    feature = np.concatenate((feature, LBPH(lbp, 25, 25)))
    return feature


def parseJAFFEName(filename):
    parts = filename.split('.')
    name, emotion = parts[0], parts[1]
    return name, emotion[:2]


directoryPath = "/home/jad/Téléchargements/jaffe"
imgsPaths = glob(directoryPath + '/*.tiff')

features = []
index = 1
JAFFE = load('../data/JAFFE')
for personName, emotion, img in JAFFE:
    a = time()
    res = [ personName, extractFeature(img), emotion ]
    features.append(res)
    print(time()-a)
    print("%d/%d"%(index, len(JAFFE)), personName, emotion)
    index = index + 1

now = datetime.now().isoformat()
print('saving file: ', 'processedJAFFE%s'%now)
save('../data/hvnLBP_JAFFE', features)
