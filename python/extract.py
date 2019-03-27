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

from storage import save

def cropFace(image):
    faceClassifier = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt.xml')
    faces = faceClassifier.detectMultiScale(image)
    #get first face coordinates
    xFace, yFace, wFace, hFace = faces[0]
    croppedFace = image[yFace:yFace+hFace, xFace:xFace+wFace]
    return croppedFace

def kirsch(image):
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
def generatePattern(matrix):
    return matrix[0, 0] + (matrix[0, 1]<<1) + (matrix[0, 2]<<2) + (matrix[1, 2]<<3) + \
           (matrix[2, 2]<<4) + (matrix[2, 1]<<5) + (matrix[2, 0]<<6) + (matrix[1, 0]<<7)

def hvnLBPWithoutBorders(image):
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
    # preprocessing
    image = cv2.equalizeHist(image)
    image = cv2.bilateralFilter(image, 5, 30, 20)
    # extract face
    image = cropFace(image)
    # rescale to 77x77 (without borders => 75x75)
    image = rescale(image, 77.0/image.shape[0], anti_aliasing=True, multichannel=False)
    filteredImages = kirsch(image)
    feature = np.array([], dtype=np.uint8)
    for _key, filteredImage in filteredImages.items():
        feature = np.concatenate((feature, LBPH(hvnLBPWithoutBorders(filteredImage), 25, 25)))
    return feature


def parseJAFFEName(filename):
    parts = filename.split('.')
    name, emotion = parts[0], parts[1]
    return name, emotion[:2]


directoryPath = "/home/jad/Téléchargements/jaffe"
imgsPaths = glob(directoryPath + '/*.tiff')

features = []
index = 1

for imgPath in imgsPaths:
    a = time()
    filename = Path(imgPath).name
    _personName, emotion = parseJAFFEName(filename)
    res = [ filename, extractFeature(cv2.imread(imgPath, 0)), emotion ]
    features.append(res)
    print(time()-a)
    print("%d/%d"%(index, len(imgsPaths)), filename, emotion)
    index = index + 1

now = datetime.now().isoformat()
print('saving file: ', 'processedJAFFE%s'%now)
save('processedJAFFE%s'%now, features)
