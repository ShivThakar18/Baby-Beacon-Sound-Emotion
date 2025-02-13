#include <stdlib.h>
#include <iostream>
#include <vector>
#include "../include/Record_BabySounds.h"
#include "../include/ADProcess_BabySounds.h"

int main(){
    //record(); 
    //std::vector<float> mfcc = extract_mfcc(ADP_FILENAME);
    //std::cout << "MFCC Features extracted\n";
    //export_mfccFile(mfcc, MFCC_FILE);
    //pyPredict_Emotions();

    const char* audio_path = "/home/Ghosttt/Baby-Beacon-Sound-Emotion/data/testing_data/belly_pain.wav";  // Change this to your file
    std::vector<float> mfccs = extract_mfcc(audio_path);

    export_mfccFile(mfccs,"/home/Ghosttt/Baby-Beacon-Sound-Emotion/output/cppOutput.txt");

    std::cout << "MFCC Features:\n";
    for (float val : mfccs) {
        std::cout << val << "\n";
    }
    std::cout << std::endl;


    //testFunction();

    return 0;

} 