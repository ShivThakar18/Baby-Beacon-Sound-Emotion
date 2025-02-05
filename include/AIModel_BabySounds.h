#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <sndfile.h>
#include "../include/ADProcess_BabySounds.h"

#define MODEL_FILENAME "../model/augmented_baby_emotions_model.h5"

int load_model(std::string model_filename);

int predict_emotion(std::vector<float> mfcc_features);