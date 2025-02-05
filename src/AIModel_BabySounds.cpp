#include "../include/AIModel_BabySounds.h"


int load_model(){
    
    cv::dnn::Net net = cv::dnn::readNet(MODEL_FILENAME);

    if(net.empty()){
        std::cerr << "Error: Could not load model " << MODEL_FILENAME << "\n";
        return -1; 
    } 
    std::cout << "Model loaded successfully!\n";
    return 0; 
}

int predict_emotion(std::vector<float> mfcc_features){
    return 0; 
}