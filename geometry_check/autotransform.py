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


class Face:
    pass


class image_analyzer:
    def __init__(self, image_path: str,
                 pretrained_predictor: str = datadir.joinpath("shape_predictor_68_face_landmarks.dat"),
                 upsample_num_times: int = 1
                 ):
        self._path = Path(image_path)
        if self._path.exists():
            self._img = cv2.imread(str(self._path.resolve()))
            if(self._img is None):
                raise ValueError(
                    "OpenCV can't read Imagefile: " + str(self._path))
        else:
            raise FileNotFoundError("Imagefile not found: " + str(self._path))
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


def main(argv):
    checks = dict({
        "top_left_corner": None,
        "top_right_corner": None,
        "bottom_left_corner": None,
        "bottom_right_corner": None,
        "transformable": False
    })

    # CHECKS STARTING HERE:
    analyzer = image_analyzer(argv[1])
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
    return checks


if __name__ == '__main__':
    check_for_needed_files(datadir)
    result = main(sys.argv)
    print(json.dumps(result))
    sys.exit(0)
