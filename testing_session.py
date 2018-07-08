# coding: utf-8
from importlib import reload
from geometry_check import autotransform
import numpy as np
import cv2
a = [i for i in range(0, 99)]
y = 33.123
img = autotransform.load_image(
    "/run/media/anaeijon/DATA/utrecht/presentation/m4075_pass.jpg")

print("size:")
print("w:", len(img[0]), "  h:", len(img))
print()

anal = autotransform.image_analyzer(img)
faces = [f for f in anal.faces()]
f = faces[0]

print("midpoint:")
print(f.midpoint())
print()

print("rotate_left(f.h_line_direction()):")
print(autotransform.rotate_left(f.h_line_direction()))
print()

corn = f.optimal_image_corners()
corn2 = np.dot(f.optimal_transform_matrix(), np.matrix.transpose(
    np.array([[p[0], p[1], 1] for p in corn])))
opt_w = int(np.ceil(np.linalg.norm(corn[1][0:2] - corn[0][0:2])))
opt_h = int(np.ceil(np.linalg.norm(corn[2][0:2] - corn[0][0:2])))
print("optimal cortners:")
print(corn)
print()
p = f.midpoint()
cv2.circle(img, (int(np.floor(p[0])), int(
    np.floor(p[1]))), 2, (0, 255, 255), 1)
for p in corn:
    cv2.circle(img, (int(np.floor(p[0])), int(
        np.floor(p[1]))), 2, (0, 255, 255), 1)
cv2.circle(img, (0, 0), 2, (0, 255, 255), 1)
cv2.imshow("Bild", img)
cv2.waitKey(1000)
while cv2.waitKey() != 27:
    print("Press ESC to exit")

# reload(autotransform)
print("Supertransformationmatrix:")
print(f.optimal_transform_matrix())
print()
print("All Corners in Image:")
print(f.optimizeable())
print()
anal = autotransform.image_analyzer(img)
faces = [f for f in anal.faces()]
f = faces[0]
M0 = np.array([np.array([1, 0, 0]), np.array([0, 1, 0])])
M0 = np.array([np.float32([1, 0, 0]), np.float32([0, 1, 0])])

cv2.imshow("Bild", cv2.warpAffine(
    # img, f.optimal_transform_matrix()[0:2], img.shape[:2]))
    img, f.optimal_transform_matrix()[0:2], img.shape[:2])[0:opt_h, 0:opt_w])
cv2.waitKey(1000)
while cv2.waitKey() != 27:
    print("Press ESC to exit")
cv2.destroyAllWindows()
