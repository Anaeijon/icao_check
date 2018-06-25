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

    def mouth_center(self):
        return np.sum(self._shape[48:60], axis=0) / 12


def main(argv):
    if len(argv) < 2:
        print("missing image file", file=sys.stderr)
        print("run: $ python3 geometry_check.py /path/to/image/file", file=sys.stderr)
        return 1

    if not hasattr(cv2, 'face'):
        print("you need to install opencv2-contrib-python", file=sys.stderr)
        print("$ pip install opencv2-contrib-python", file=sys.stderr)
        return 1

    checks = dict({
        "ratio_correct": False,
        "face_detected": False,
        "single_face": False,
        "h_line_almost_horizontal": False,
        "h_line_rotation": 1,
        "v_line_almost_vertical": False,
        "v_line_rotation": 1,
        "midpoint_in_vertical_center": False,
        "midpoint_in_upper_half": False,
        "midpoint": None,
        "head_width_correct": False,
        "head_width_ratio": 1
    })

    # CHECKS STARTING HERE:
    analyzer = image_analyzer(argv[1])
    checks["ratio_correct"] = bool(analyzer.check_image_ratio(min=0.74, max=0.80))
    faces = [f for f in analyzer.faces()]
    checks["face_detected"] = bool(len(faces) > 0)
    if len(faces) == 1:
        checks["single_face"] = True
        face = faces[0]
        #   I: H line almost horizontal:
        #      (f.eye_left()-f.eye_right())[1] rund 0
        #          abs( -||- / ana.height() ) < 0.01 # weniger als 1% rotation
        h_rotation = (face.eye_left() - face.eye_right()
                      )[1] / analyzer.height()
        checks["h_line_almost_horizontal"] = bool(abs(h_rotation) < 0.01)
        checks["h_line_rotation"] = float(h_rotation)

        #  II: V line almost vertical,
        #      H line is more important, because V and H don't need to be perpendicular
        #       (f.eye_center()-f.mouth_center())[0] rund 0
        #          abs( -||- / ana.width() ) < 0.05 # weniger als 5% rotation
        v_rotation = (face.eye_center() - face.mouth_center())[0] / analyzer.width()
        checks["v_line_almost_vertical"] = bool(abs(v_rotation) < 0.05)
        checks["v_line_rotation"] = float(v_rotation)

        # III: Midpoint M needs to be in horizontal center and vertically 30%-50% from top
        #      f.eye_center()/[ana.width(),ana.height()] = [ 0.45 <= x <= 0.55 , 0.30 <= y <= 0.50 ]
        m_rel = face.eye_center() / [analyzer.width(), analyzer.height()]
        checks["midpoint_in_vertical_center"] = bool((m_rel[0] >= 0.45) and (m_rel[0] <= 0.55))
        checks["midpoint_in_upper_half"] = bool((m_rel[1] >= 0.30) and (m_rel[1] <= 0.50))
        checks["midpoint"] = [float(d) for d in m_rel]

        #  IV: Headwith ratio
        head_width_ratio = face.rect().width() / analyzer.width()
        checks["head_width_correct"] = bool((head_width_ratio >= 0.5) and (head_width_ratio <= 0.75))
        checks["head_width_ratio"] = float(head_width_ratio)

        if show_rectangle:
            img = analyzer.img()
            cv2.namedWindow("Bild", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Bild", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            rect = face.rect()
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()),  (0, 255, 255))
            cv2.imshow("Bild", img)
            while(1):
                k = cv2.waitKey(33)
                if k == 27 or k == ord('q'):    # Esc key or q to stop
                    cv2.destroyAllWindows()
                    break

    return checks


if __name__ == '__main__':
    check_for_needed_files(datadir)
    result = main(sys.argv)
    print(json.dumps(result))
    sys.exit(0)
