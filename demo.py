import cv2
import pdb

from nnocr import mser
from nnocr import model
from nnocr import classifier

img = cv2.imread('./samples/phone1.jpg')

(regions,bboxes) = mser.get_mser(img)

model = model.get_char_classifier()

classifier.get_chars(img,model,regions,bboxes)
