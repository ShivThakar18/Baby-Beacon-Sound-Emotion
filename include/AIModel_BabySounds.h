#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <sndfile.h>
#include <string>
#include "../include/ADProcess_BabySounds.h"

#define MODEL_FILENAME "/home/ghosttt/Baby-Beacon-Sound-Emotion/model/frozen_model.pb"


int load_model();

int predict_emotion(std::vector<float> mfcc_features);