""" extract features
"""
from time import time
import cv2
import numpy as np
from numba import jit
from skimage.feature import local_binary_pattern
from skimage.transform import rescale

from storage import save, load

def cropFace(image):
    """ crop face from a given image
    """
    fc = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_alt.xml')
    faces = fc.detectMultiScale(image)
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

# pylint: disable=invalid-name
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
def TLPPixel (centre, pixel, threshold):
    """ compute neighboor value for upper and lower
    """
    # -1
    if pixel + threshold < centre:
        return np.array([0, 1], dtype=np.uint8)
    # 1
    elif pixel > threshold + centre:
        return np.array([1, 0], dtype=np.uint8)
    # 0
    else:
        return np.array([0, 0], dtype=np.uint8)

@jit(nopython=True)
def TLPBloc(arr, t):
    """ compute pixel value ternary local pattern
    """
    centrePixel = arr[1, 1]
    tlpup = np.empty_like(arr)
    tlplo = np.empty_like(arr)
    tlpup[0, 0], tlplo[0, 0] = TLPPixel(centrePixel, arr[0, 0], threshold=t)
    tlpup[0, 1], tlplo[0, 1] = TLPPixel(centrePixel, arr[0, 1], threshold=t)
    tlpup[0, 2], tlplo[0, 2] = TLPPixel(centrePixel, arr[0, 2], threshold=t)
    tlpup[1, 2], tlplo[1, 2] = TLPPixel(centrePixel, arr[1, 2], threshold=t)
    tlpup[2, 2], tlplo[2, 2] = TLPPixel(centrePixel, arr[2, 2], threshold=t)
    tlpup[2, 1], tlplo[2, 1] = TLPPixel(centrePixel, arr[2, 1], threshold=t)
    tlpup[2, 0], tlplo[2, 0] = TLPPixel(centrePixel, arr[2, 0], threshold=t)
    tlpup[1, 0], tlplo[1, 0] = TLPPixel(centrePixel, arr[1, 0], threshold=t)
    return np.array([generatePattern(tlpup), generatePattern(tlplo)], np.uint8)

@jit(nopython=True)
def TLP(image, threshold):
    """ generate the upper and lower images of ternary local pattern
    """
    upper = np.empty_like(image)
    lower = np.empty_like(image)
    height = image.shape[0]
    width = image.shape[1]
    for w in range(1, width - 1):
        for h in range(1, height - 1):
            subImg = image[h-1:h+2, w-1:w+2]
            upper[h, w], lower[h, w] = TLPBloc(subImg, threshold)
    return [upper, lower]

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

def LBPH(image, gridX, gridY):
    """ generate LBP histogram
    """
    cellWidth = image.shape[1] // gridX
    cellHeight = image.shape[0] // gridY
    
    feature = np.array(())
    for x in range(0, gridX*cellWidth, cellWidth):
        for y in range(0, gridY*cellHeight, cellHeight):
            # extract cell
            subImg = image[y:y+cellHeight, x:x+cellWidth]
            # construct histogram
            hist, _binEdge = np.histogram(subImg.ravel(), bins=256, range=[0, 255],
                                          density=True)
            # concatenate histograms
            feature = np.concatenate((feature, hist))
    return feature

# pylint: disable=dangerous-default-value, comparison-with-callable
def extractFeature(image, numBiFilter=1, kirschFilter=False, method=None, methodArgs={},
                   gridX=18, gridY=18, rescaleImage=False):
    """ extract features from image
    """
    feature = np.array(())
    # preprocessing
    image = cv2.equalizeHist(image)
    for _ in range(numBiFilter):
        image = cv2.bilateralFilter(image, 5, 30, 20)
    # extract face
    image = cropFace(image)
    
    if rescaleImage:
        # rescale to 77x77 if method used is hvnLBP (without borders => 75x75)
        targetResolution = 77.0 if method==hvnLBPWithoutBorders else 75.0
        image = rescale(image, targetResolution/image.shape[0], anti_aliasing=True,
                        multichannel=False, mode='constant')
    if kirschFilter:
        filteredImages = kirsch(image)
        for filteredImage in filteredImages.values():
            lbpImage = method(filteredImage, **methodArgs)
            if isinstance(lbpImage, list):
                for lbpImg in lbpImage:
                    feature = np.concatenate((feature, LBPH(lbpImg, gridX, gridY)))
            else:
                feature = np.concatenate((feature, LBPH(method(lbpImage, **methodArgs),
                                                        gridX, gridY)))
    else:
        lbpImage = method(image, **methodArgs)
        if isinstance(lbpImage, list):
            for lbpImg in lbpImage:
                feature = np.concatenate((feature, LBPH(lbpImg, gridX, gridY)))
        else:
            feature = LBPH(lbpImage, gridX, gridY)
    return feature

features = []
index = 1
JAFFE = load('../data/JAFFE')
jaffeLength = len(JAFFE)
for personName, emotion, img in JAFFE:
    a = time()
    params = { "angle": 0 }
    extractedFeature = extractFeature(img,
                                      numBiFilter=1,
                                      rescaleImage=False,
                                      kirschFilter=True,
                                      method=dLBP,
                                      methodArgs=params,
                                      gridX=18,
                                      gridY=18)
    res = [ personName, extractedFeature, emotion ]
    features.append(res)
    print("%d/%d"%(index, jaffeLength), personName, emotion,  time()-a)
    index = index + 1

filename = 'dLBP90_JAFFE1'

print('saving file: ', '/data/%s'%filename)
save('../data/%s'%filename, features)
