#!./icao-venv/bin/python3

# needs:    opencv2-python
#           opencv2-contrib-python
#           dlib

import os
from pathlib import Path
import sys
import cv2
import dlib
import numpy as np
import json


show_rectangle = False

# set here, where to look for pretrained models:
datadir = Path("data")

# following files should be present in the datadir:
needed_files = {
    "shape_predictor_68_face_landmarks.dat":
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
}


def check_for_needed_files(datadir: Path):
    from urllib import request

    if not datadir.is_dir():
        datadir.mkdir()

    # Downloading needed Files
    for filename, url in needed_files.items():
        file = datadir.joinpath(filename)
        newfile = dlfile = file.parent.joinpath(Path(url).name)

        if not file.exists():
            print(
                "Downloading  " + url,
                file=sys.stderr
            )
            request.urlretrieve(
                url,
                filename=dlfile,
                reporthook=lambda blocks, blocksize, fullsize:
                    print(str((blocks * blocksize * 100) /
                              fullsize)[:5] + "%" + "\033[F", file=sys.stderr)
            )
            print("Done" + (" " * 12), file=sys.stderr)

            # decompress BZ2 files
            if dlfile.suffix == ".bz2":
                import bz2
                newfile = dlfile.parent.joinpath(dlfile.stem)
                if not newfile.exists():
                    print("Decompressing " + str(dlfile), file=sys.stderr)
                    newfile.write_bytes(bz2.decompress(dlfile.read_bytes()))
                    dlfile.unlink()

            # mv file to desired name
            if newfile.resolve() != file.resolve():
                newfile.rename(file)


# origin vectors o1, o2
# direction vectors d1, d2 (normalized)
def line_intersection(o1, d1, o2, d2):
    u = (o1[1] * d2[0] + d2[1] * o2[0] - o2[1] * d2[0] -
         d2[1] * o1[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
    p = o1 + d1 * u
    return(p)


# rotate a vector counter clockwise
def rotate_left(v):
    return np.array([-v[1], v[0]])


def load_image(image_path: str):
    _path = Path(image_path)
    if _path.exists():
        _img = cv2.imread(str(_path.resolve()))
        if(_img is None):
            raise ValueError(
                "OpenCV can't read Imagefile: " + str(_path))
            return None
    else:
        raise FileNotFoundError("Imagefile not found: " + str(_path))
        return None
    return _img


class image_analyzer:
    def __init__(self, image,
                 pretrained_predictor: str = datadir.joinpath(
                     "shape_predictor_68_face_landmarks.dat"),
                 upsample_num_times: int = 1
                 ):
        self._img = image
        self._pretrained_predictor = Path(pretrained_predictor)
        self._upsample_num_times = upsample_num_times

    def img(self):
        return self._img

    def width(self):
        return len(self._img[0])

    def height(self):
        return len(self._img)

    def image_ratio(self):
        return len(self._img[0]) / len(self._img)

    def check_image_ratio(self, min: float=0.74, max: float=0.80):
        return (min <= self.image_ratio() <= max)

    def face_detector(self):
        # get static face detector
        if not hasattr(image_analyzer, '_face_detector') or image_analyzer._face_detector is None:
            image_analyzer._face_detector = dlib.get_frontal_face_detector()
        return image_analyzer._face_detector

    def shape_predictor(self):
        # load static shape predictor:
        if not hasattr(self, '_shape_predictor') or self._shape_predictor is None:
            self._shape_predictor = dlib.shape_predictor(
                str(self._pretrained_predictor)
            )
        return self._shape_predictor

    def face_rectangles(self):
        # check if image changed:
        if not hasattr(self, '_face_rectangles') or self._face_rectangles is None:
            self._face_rectangles = self.face_detector()(
                image=self._img,
                upsample_num_times=self._upsample_num_times
            )
        return self._face_rectangles

    def shape_of_face(self, face: dlib.full_object_detection):
        return self.shape_predictor()(self._img, face)

    def shapes(self):
        # generator for facial landmarks
        for face in self.face_rectangles():
            yield self.shape_of_face(face)

    def faces(self):
        for face in self.face_rectangles():
            yield Face(face, self)


class Face:
    def __init__(self,
                 rectangle: dlib.full_object_detection,
                 analyzer: image_analyzer):
        self._rect = rectangle
        self._img = analyzer.img()
        self._shape = [np.array([p.x, p.y])
                       for p in analyzer.shape_of_face(rectangle).parts()]

    def shape(self):
        return self._shape

    def rect(self):
        return self._rect

    def eye_left(self):
        return (self._shape[36] + self._shape[39]) / 2

    def eye_right(self):
        return (self._shape[42] + self._shape[45]) / 2

    def eye_center(self):
        return (self.eye_left() + self.eye_right()) / 2

    def chin(self):
        return self._shape[8] * 1

    def ear_left(self):
        return self._shape[0] * 1

    def ear_right(self):
        return self._shape[16] * 1

    def midpoint(self):
        return self.eye_center()

    def mouth_center(self):
        return np.sum(self._shape[48:60], axis=0) / 12

    def h_line_direction(self):
        h = self.eye_right() - self.eye_center()
        return h / np.linalg.norm(h)

    def v_line_direction(self):
        v = self.mouth_center() - self.eye_center()
        return v / np.linalg.norm(v)

    def left_eye_distance(self):
        return np.linalg.norm(self.eye_left() - self.eye_center())

    def right_eye_distance(self):
        return np.linalg.norm(self.eye_right() - self.eye_center())

    def chin_distance(self):
        return np.linalg.norm(self.chin() - self.eye_center())

    def bottom_left_corner(self):
        chin2 = line_intersection(self.chin(),
                                  self.h_line_direction(),
                                  self.eye_center(),
                                  rotate_left(self.h_line_direction()))
        ear_l = line_intersection(self.ear_left(),
                                  rotate_left(self.h_line_direction()),
                                  self.eye_center(),
                                  self.h_line_direction())
        return line_intersection(chin2,
                                 self.h_line_direction(),
                                 ear_l,
                                 rotate_left(self.h_line_direction()))

    def bottom_right_corner(self):
        chin2 = line_intersection(self.chin(),
                                  self.h_line_direction(),
                                  self.eye_center(),
                                  rotate_left(self.h_line_direction()))
        ear_r = line_intersection(self.ear_right(),
                                  rotate_left(self.h_line_direction()),
                                  self.eye_center(),
                                  self.h_line_direction())
        return line_intersection(chin2,
                                 self.h_line_direction(),
                                 ear_r,
                                 rotate_left(self.h_line_direction()))

    def top_left_corner(self):
        chin2 = line_intersection(self.chin(),
                                  self.h_line_direction(),
                                  self.eye_center(),
                                  rotate_left(self.h_line_direction()))
        fake_forehead = self.eye_center() + (self.eye_center() - chin2)
        ear_l = line_intersection(self.ear_left(),
                                  rotate_left(self.h_line_direction()),
                                  self.eye_center(),
                                  self.h_line_direction())
        return line_intersection(fake_forehead,
                                 self.h_line_direction(),
                                 ear_l,
                                 rotate_left(self.h_line_direction()))

    def top_right_corner(self):
        chin2 = line_intersection(self.chin(),
                                  self.h_line_direction(),
                                  self.eye_center(),
                                  rotate_left(self.h_line_direction()))
        fake_forehead = self.eye_center() + (self.eye_center() - chin2)
        ear_r = line_intersection(self.ear_right(),
                                  rotate_left(self.h_line_direction()),
                                  self.eye_center(),
                                  self.h_line_direction())
        return line_intersection(fake_forehead,
                                 self.h_line_direction(),
                                 ear_r,
                                 rotate_left(self.h_line_direction()))

    def face_width(self):
        return np.linalg.norm(self.top_right_corner() - self.top_left_corner())

    def face_height(self):
        return np.linalg.norm(self.top_left_corner() - self.bottom_left_corner())

    def rotation_matrix(self):
        # returns a rotation matrix which will rotate h parallel to x=(1,0)
        h = self.h_line_direction()
        return [[h[0],  h[1], 0],
                [-h[1], h[0], 0],
                [0,     0,    1]]

    def translation_to_0_matrix(self):
        # returns a translation matrix, which will move midpoint to (0,0)
        m = self.eye_center()
        return [[1, 0, -m[0]],
                [0, 1, -m[1]],
                [0, 0,     1]]

    def supertransformationmatrix(self,
                                  side_distance: float = (3 / 16),
                                  midpoint_top_distance: float = (5 / 12),
                                  image_ratio: float = (7 / 9)):
        first_trans = np.dot(self.rotation_matrix(),
                             self.translation_to_0_matrix())
        cs = self.optimal_image_corners(side_distance=side_distance,
                                        midpoint_top_distance=midpoint_top_distance,
                                        image_ratio=image_ratio)
        trans_tl = np.dot(first_trans, np.append(cs[0], 1))
        final_trans = [[1, 0, -trans_tl[0]],
                       [0, 1, -trans_tl[1]],
                       [0, 0,            1]]
        return np.dot(final_trans, first_trans)

    def optimal_image_corners(self,
                              side_distance: float = (3 / 16),
                              midpoint_top_distance: float = (5 / 12),
                              image_ratio: float = (7 / 9)):
        # w = head width
        w = self.face_width()
        # a = image width
        # image ratio:
        #        0.5  <= w/a <=  0.75
        #       (1/2) <= w/a <= (3/4)
        # optimal w/a:
        #       ((1/2)+(3/4))/2 = 5/8 = 0.625
        # optimal distance from ear to image border (side_distance):
        #       (1-(5/8))/2 = 3/16
        a = w * (1 + 2 * side_distance)
        # optimal picture ratio:
        #   common professional picture ratio is 35mm/45mm
        #   so assume (7/9) = 0.77777777
        #   0.74 and 0.80 are ( 0.77 [+|-] 0.03 )
        #   assuming:
        #   (7/9)-(1/30) and (7/9)+(1/30)
        m_to_bottom = 1 - midpoint_top_distance

        # b = image height
        b = w / image_ratio
        leftest_on_h = self.midpoint() - self.h_line_direction() * (2 / a)
        rightest_on_h = self.midpoint() + self.h_line_direction() * (2 / a)
        bottom_on_ht = self.midpoint() - rotate_left(self.h_line_direction()) * m_to_bottom
        top_on_ht = bottom_on_ht + rotate_left(self.h_line_direction()) * b

        return [
            line_intersection(
                leftest_on_h,
                rotate_left(self.h_line_direction()),
                top_on_ht,
                self.h_line_direction()),
            line_intersection(
                rightest_on_h,
                rotate_left(self.h_line_direction()),
                top_on_ht,
                self.h_line_direction()),
            line_intersection(
                leftest_on_h,
                rotate_left(self.h_line_direction()),
                bottom_on_ht,
                self.h_line_direction()),
            line_intersection(
                rightest_on_h,
                rotate_left(self.h_line_direction()),
                bottom_on_ht,
                self.h_line_direction())
        ]


def main(argv):
    checks = dict({
        "top_left_corner": None,
        "top_right_corner": None,
        "bottom_left_corner": None,
        "bottom_right_corner": None,
        "transformable": False
    })

    # CHECKS STARTING HERE:
    analyzer = image_analyzer(load_image(argv[1]))
    faces = [f for f in analyzer.faces()]
    if len(faces) == 1:
        face = faces[0]
        tl = face.top_left_corner()
        tr = face.top_right_corner()
        bl = face.bottom_left_corner()
        br = face.bottom_right_corner()
        checks["top_left_corner"] = [float(d) for d in tl]
        checks["top_right_corner"] = [float(d) for d in tr]
        checks["bottom_left_corner"] = [float(d) for d in bl]
        checks["bottom_right_corner"] = [float(d) for d in br]
        checks["transformable"] = bool(
            0 <= tl[0] <= analyzer.width() and
            0 <= tl[1] <= analyzer.height() and
            0 <= tr[0] <= analyzer.width() and
            0 <= tr[1] <= analyzer.height() and
            0 <= bl[0] <= analyzer.width() and
            0 <= bl[1] <= analyzer.height() and
            0 <= br[0] <= analyzer.width() and
            0 <= br[1] <= analyzer.height())
        if checks["transformable"]:
            pass
    return checks


if __name__ == '__main__':
    check_for_needed_files(datadir)
    result = main(sys.argv)
    print(json.dumps(result))
    sys.exit(0)
