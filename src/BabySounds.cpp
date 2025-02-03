#include <opencv4/opencv2/opencv.hpp> // import opencv files 
#include <stdlib.h>
#include <iostream>
#include <portaudio.h>
#include <sndfile.h>
#include <vector>

/*
 * To compile and run: 
 *      > g++ src/BabySounds.cpp -o BabySounds `pkg-config --cflags --libs opencv4`
 *      > ./BabySounds
 */ 

int main(){
    std::cout << "Hello World!"<< std::endl;
    std::cout << "OpenCV Version: "<< CV_VERSION << std::endl; // check if OpenCV has imported correctly
    
}
