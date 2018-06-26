#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include <iostream>

using namespace dlib;

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cout << "Example:" << std::endl;
            std::cout << "./neutralface.run shape_predictor_68_face_landmarks.dat path/photo.jpg" << std::endl;
            std::cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:" << std::endl;
            std::cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
            return 0;
        }

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

        array2d<rgb_pixel> img;
        load_image(img, argv[2]);
        pyramid_up(img);

        std::vector<rectangle> dets = detector(img);
        if(dets.size() > 0) { //crash when no pics available
          full_object_detection shape = sp(img, dets[0]);

          bool bSmiling = false;
          bool bOpenmouth = false;
          
          if((shape.part(67).y() - shape.part(61).y()) > 20 ||
             (shape.part(66).y() - shape.part(62).y()) > 20 ||
             (shape.part(65).y() - shape.part(63).y()) > 20) 
              bOpenmouth = true;
          if((shape.part(62).y() - shape.part(60).y()) > 20 &&
             (shape.part(62).y() - shape.part(64).y()) > 20 &&
             (shape.part(62).y() - shape.part(48).y()) > 25 &&
             (shape.part(62).y() - shape.part(54).y()) > 25 &&
             ((shape.part(60).y() - shape.part(48).y()) > 5 ||
             (shape.part(64).y() - shape.part(54).y()) > 5))  
              bSmiling = true;

          bool bNeutral = false;
          if(!bSmiling && !bOpenmouth) bNeutral = true;

          std::cout << "{\"not_smiling\":" << (bSmiling?"false":"true") << ",";
          std::cout << "\"shut_mouth\":" << (bOpenmouth?"false":"true") << ",";
          std::cout << "\"neutral_expression\":" << (bNeutral?"true":"false") << "}";
        } else {
          std::cout << "{}";
        }

    } catch (std::exception& e) {
        std::cout << "\nexception: " << e.what() << std::endl;
    }
}