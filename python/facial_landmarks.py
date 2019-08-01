""" Extract facial landmark features
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
                                           device='cpu',
                                           verbose=False)
        
    @staticmethod
    def _polyArea(points):
        """ calculate the area of a polygon (Gauss equation)
        """
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    @staticmethod
    def _slope(point1, point2):
        """ compute the slope of the line passing through two points
        """
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    
    def landmarks(self, img):
        """ get landmarks
        """
        return self.faceAlignment.get_landmarks(img)[0]
        
    def geometricFeatures1(self, img, landmarks=None):
        """Improved Performance in Facial Expression Recognition Using 32 Geometric Features
            Linear features(15):
                – 3 for left eyebrow
                – 2 for left eye
                – 1 for cheeks
                – 1 for nose
                – 8 for mouth
            Polygonal features(3):
                – 1 for the left eye
                – 1 between corners of left eye and left corner of mouth
                – 1 for mouth
            Elliptical features(7):
                – 1 for left eyebrow
                – 3 for the left eye: eye, upper and lower eyelids
                – 3 for mouth: upper and lower lips
            Slope features(7):
                – 1 for left eyebrow
                – 6 for mouth corners
        """
        pts = self.faceAlignment.get_landmarks(img)[0] if landmarks is None else landmarks
        result = []
        # Linear features
        #eyebrow
        result.append(d(pts[21], pts[22]))
        result.append(d(pts[22], pts[42]))
        result.append(d(pts[26], pts[45]))
        #left eye
        result.append(d(pts[42], pts[45]))
        #d2=43-47 || 44-46
        result.append(d(pts[43], pts[47]))
        
        #cheeks
        result.append(d(pts[1], pts[15]))
        
        #nose
        result.append(d(pts[31], pts[35]))
        
        #mouth
        result.append(d(pts[48], pts[51]))
        result.append(d(pts[51], pts[54]))
        result.append(d(pts[62], pts[66]))
        result.append(d(pts[51], pts[57]))
        result.append(d(pts[50], pts[33]))
        result.append(d(pts[52], pts[33]))
        result.append(d(pts[48], pts[31]))
        result.append(d(pts[54], pts[35]))
        
        #Polygonal features:
        result.append(FaceTools._polyArea([pts[48], pts[54], pts[57]]))
        result.append(FaceTools._polyArea([pts[54], pts[42], pts[45]]))
        result.append(FaceTools._polyArea([pts[42], pts[22], pts[26], pts[45]]))
        
        #Slope features:
        result.append(FaceTools._slope(pts[22], pts[26]))
        result.append(FaceTools._slope(pts[48], pts[31]))
        result.append(FaceTools._slope(pts[54], pts[35]))
        result.append(FaceTools._slope(pts[48], pts[51]))
        result.append(FaceTools._slope(pts[51], pts[54]))
        result.append(FaceTools._slope(pts[54], pts[57]))
        result.append(FaceTools._slope(pts[48], pts[57]))
        
        return result
    
    def faceDistances(self, img, landmarks=None):
        """ calculate distances as described in
            'Automatic Facial Expression Recognition Using Combined Geometric Features':
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
    
    