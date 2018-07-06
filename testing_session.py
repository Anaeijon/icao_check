# coding: utf-8
from importlib import reload
from geometry_check import autotransform
import numpy as np
import cv2
a = [i for i in range(0,99)]
y = 33.123
img = autotransform.load_image("/run/media/anaeijon/DATA/utrecht/presentation/m4075_pass.jpg")
anal = autotransform.image_analyzer(img)
faces = [f for f in anal.faces()]
f = faces[0]
reload(autotransform)
f.supertransformationmatrix()
f.supertransformationmatrix()
anal = autotransform.image_analyzer(img)
faces = [f for f in anal.faces()]
f = faces[0]
M0 = np.array([np.array([1,0,0]),np.array([0,1,0])])
M0 = np.array([np.float32([1,0,0]),np.float32([0,1,0])])
cv2.imshow("Bild",cv2.warpAffine(img, f.supertransformationmatrix()[0:2], img.shape[:2]))
cv2.waitKey(1000)
while cv2.waitKey() != 27:
    print( "Press ESC to exit")
cv2.destroyAllWindows()
