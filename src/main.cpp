#include <stdlib.h>
#include <iostream>
#include <vector>
#include "../include/Record_BabySounds.h"
#include "../include/ADProcess_BabySounds.h"
#include <filesystem>
#include <string>
namespace fs = std::filesystem;
int main(){
    //record(); 
    //std::vector<float> mfcc = extract_mfcc(ADP_FILENAME);
    //std::cout << "MFCC Features extracted\n";
    //export_mfccFile(mfcc, MFCC_FILE);
    //pyPredict_Emotions();

    std::string emotion = "tired";
    const char* DIR = "/home/Ghosttt/Baby-Beacon-Sound-Emotion/data/dataset/augmented_dataset/tired";

    // Check if the directory exists
    if (!fs::exists(DIR) || !fs::is_directory(DIR)) {
        std::cerr << "Directory does not exist or is not a directory!" << std::endl;
        return 1;
    }

    int counter = 0; 
    // Iterate through the directory and process each file
    for (const auto& entry : fs::directory_iterator(DIR)) {
        if (fs::is_regular_file(entry)) { // Ensure it's a file, not a directory
            std::vector<float> features = extract_mfcc(entry.path().c_str()); // Convert fs::path to const char*
            
            std::ofstream FILE("/home/Ghosttt/Baby-Beacon-Sound-Emotion/output/"+emotion+"/"+ emotion + std::to_string(counter) + ".txt");
            if(!FILE){
                std::cerr << "Error: Could not open the file for writing\n";
                return 0; 
            }

            for(const float& value : features){
                FILE << value << "\n";
            }

            FILE.close();
            std::cout << "MFCCs Features written to "<< emotion << std::to_string(counter) <<".txt\n";
            counter = counter + 1; 

        }
    }


    //const char* audio_path = "/home/Ghosttt/Baby-Beacon-Sound-Emotion/data/testing_data/belly_pain.wav";  // Change this to your file
    //std::vector<float> mfccs = extract_mfcc(audio_path);

    //export_mfccFile(mfccs,"/home/Ghosttt/Baby-Beacon-Sound-Emotion/output/cppOutput.txt");

    //testFunction();

    return 0;

} 