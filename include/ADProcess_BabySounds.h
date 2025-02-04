#include <iostream>
#include <vector>
#include <aubio/aubio.h>
#include <sndfile.h>

#define MFCC_SAMPLE_RATE 22050
#define N_FILTERS 40
#define N_MFCC 13
#define FRAME_SIZE 512
#define HOP_SIZE 256
#define ADP_FILENAME "recording.wav"

std::vector<float> extract_mfcc(const char* filename);
