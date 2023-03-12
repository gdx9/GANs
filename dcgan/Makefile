CXX = g++
CXXFLAGS = -std=c++17 -Wall

INC_DIRS = -I. -I/usr/local/include/opencv4
LIB_DIRS = -L. -L/usr/local/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_ml -lopencv_flann -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_videoio -lopencv_dnn

CHECK_GENERATOR = dcgan_check
CHECK_GENERATOR_OBJ = dcgan_check.o

all: $(CHECK_GENERATOR)

$(CHECK_GENERATOR_OBJ): dcgan_check.cpp
	$(CXX) $(CXXFLAGS) -c $< $(INC_DIRS) -o $@

$(CHECK_GENERATOR): $(CHECK_GENERATOR_OBJ)
	$(CXX) $(CXXFLAGS) $(LIB_DIRS) -o $@ $^ $(OPENCV_LIBS)

clean:
	rm -f $(CHECK_GENERATOR) $(CHECK_GENERATOR_OBJ)
