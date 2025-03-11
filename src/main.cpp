#include <stdlib.h>
#include <iostream>
#include <vector>
#include "../include/Record_BabySounds.h"
#include "../include/ADProcess_BabySounds.h"
#include <filesystem>
#include <string>
namespace fs = std::filesystem;
int main(){
    
    record(); 
    std::vector<float> mfcc = extract_mfcc(ADP_FILENAME);
    std::cout << "MFCC Features extracted\n";
    export_mfccFile(mfcc, MFCC_FILE);

    return 0;

} 