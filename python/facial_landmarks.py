"""
"""
import numpy as np
from scipy.spatial.distance import euclidean as d
from face_alignment import FaceAlignment, LandmarksType

class FaceTools:
    """ Face Toolkit
    """
    def __init__(self, dimensions='2d'):
        landmarkType = LandmarksType._2D if dimensions == '2d' else LandmarksType._3D
        self.faceAlignment = FaceAlignment(landmarkType,
                                           flip_input=False,
                                           device='cuda',
                                           verbose=False)
        
    @staticmethod
    def polyArea(x, y):
        """ calculate the area of a polygon
        """
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    def landmarks(self, img):
        """ get landmarks
        """
        return self.faceAlignment.get_landmarks(img)[0]
    
    def faceDistances(self, img, landmarks=None):
        """ calculate distances:
            D1  Left eyebrow length
            D2  Right eyebrow length
            D3  Distance between left and right eyebrow
            D4  Left eye height
            D5  Left eye width
            D6  Right eye height
            D7  Right eye width
            D8  Distance between left eyebrow and left eye
            D9  Distance between right eyebrow and right eye
            D10 Distance between nose tip and upper lip
            D11 Lip width
            D12 Lip height
            D13 Inner lip distance
            D14 Distance between left eye corner and lip left corner
            D15 Distance between right eye corner and lip right corner
        """
        pts = self.faceAlignment.get_landmarks(img)[0] if landmarks is None else landmarks
        distances = [
            d(pts[17], pts[18]) + d(pts[18], pts[19]) + d(pts[19], pts[20]) + \
            d(pts[20], pts[21]),
            d(pts[22], pts[23]) + d(pts[23], pts[24]) + d(pts[24], pts[25]) + \
            d(pts[25], pts[26]),
            d(pts[21], pts[22]),
            d(pts[40], pts[38]),
            d(pts[36], pts[39]),
            d(pts[43], pts[47]),
            d(pts[42], pts[45]),
            d(pts[19], pts[37]),
            d(pts[24], pts[44]),
            d(pts[33], pts[51]),
            d(pts[48], pts[54]),
            d(pts[51], pts[57]),
            d(pts[62], pts[66]),
            d(pts[36], pts[48]),
            d(pts[45], pts[54])
        ]
        
        return distances
    
    