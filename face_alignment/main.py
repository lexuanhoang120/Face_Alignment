# import the necessary packages


import cv2
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

# # construct the argument parser and parse the arguments
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
path_shape_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(path_shape_predictor)
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
path_image = 'photo_2022-12-23_17-04-55.jpg'
image = cv2.imread(path_image)
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
cv2.waitKey(0)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    x, y, w, h = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)
    # display the output images
cv2.imshow("Original", faceOrig)
cv2.imshow("Aligned", faceAligned)
cv2.waitKey(0)
