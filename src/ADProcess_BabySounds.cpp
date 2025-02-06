#include "../include/ADProcess_BabySounds.h"

/* std::vector<float> extract_mfcc(const char* filename){

    std::vector<float> mfcc_features(N_MFCC,0.0);

    // OPEN WAV FILE
    SF_INFO sfinfo;
    SNDFILE* FILE = sf_open(filename,SFM_READ,&sfinfo);

    if(!FILE){
        std::cerr << "Error opening audio file!\n";
        return mfcc_features; 
    }

    // Aubio Buffers
    fvec_t* input = new_fvec(FRAME_SIZE); 
    cvec_t* fft_output = new_cvec(FRAME_SIZE);
    aubio_fft_t* fft = new_aubio_fft(FRAME_SIZE);                                 // Fast Fourier Transform
    aubio_mfcc_t* mfcc = new_aubio_mfcc(FRAME_SIZE,N_FILTERS,N_MFCC,MFCC_SAMPLE_RATE); // MFCC Buffer

    std::vector<float> mfcc_sum(N_MFCC,0.0);
    int frame_count = 0; 

    // Process Audio 
    while(sf_read_float(FILE, input->data, FRAME_SIZE) > 0){
        // Compute the Fourier Transform
        aubio_fft_do(fft, input,fft_output);

        // Compute MFCCs
        fvec_t* mfcc_out = new_fvec(N_MFCC);
        aubio_mfcc_do(mfcc,fft_output,mfcc_out);

        // Accumulate MFCCs for averaging
        int i = 0;
        for(i = 0; i < N_MFCC; i++){
            mfcc_sum[i] += mfcc_out -> data[i];
        }

        del_fvec(mfcc_out);
        frame_count++;      // increment frames
    }

    if(frame_count > 0){
        int i = 0; 
        for(i = 0; i < N_MFCC; i++){
            mfcc_features[i] = mfcc_sum[i] / frame_count; // compute mean MFCC values
        }
    }

    sf_close(FILE);
    del_aubio_fft(fft);
    del_cvec(fft_output);
    del_aubio_mfcc(mfcc);
    del_fvec(input);

    return mfcc_features;  // return extracted features

} */

std::vector<float> extract_mfcc(const char* filename){
    std::vector<float> mfcc_features(N_MFCC, 0.0);

    // Open WAV file
    SF_INFO sfinfo;
    SNDFILE* file = sf_open(filename, SFM_READ, &sfinfo);
    if (!file) {
        std::cerr << "Error opening audio file!\n";
        return mfcc_features;
    }

    // Initialize Aubio buffers
    fvec_t* input = new_fvec(FRAME_SIZE);
    cvec_t* fft_output = new_cvec(FRAME_SIZE);
    aubio_fft_t* fft = new_aubio_fft(FRAME_SIZE);
    aubio_mfcc_t* mfcc = new_aubio_mfcc(FRAME_SIZE, N_FILTERS, N_MFCC, MFCC_SAMPLE_RATE);

    std::vector<float> mfcc_sum(N_MFCC, 0.0);
    int frame_count = 0;

    // Read audio and process frames
    while (sf_read_float(file, input->data, FRAME_SIZE) > 0) {
        // Apply pre-emphasis filter (to match Librosa)
        for (int i = FRAME_SIZE - 1; i > 0; i--) {
            input->data[i] -= 0.97 * input->data[i - 1];
        }

        // Compute FFT
        aubio_fft_do(fft, input, fft_output);

        // Compute MFCCs
        fvec_t* mfcc_out = new_fvec(N_MFCC);
        aubio_mfcc_do(mfcc, fft_output, mfcc_out);

        // Accumulate MFCCs for averaging
        for (int i = 0; i < N_MFCC; i++) {
            mfcc_sum[i] += mfcc_out->data[i];
        }

        del_fvec(mfcc_out);
        frame_count++;
    }

    // Compute the mean MFCC values
    if (frame_count > 0) {
        for (int i = 0; i < N_MFCC; i++) {
            mfcc_features[i] = mfcc_sum[i] / frame_count;
        }
    }

    // Cleanup
    sf_close(file);
    del_aubio_fft(fft);
    del_cvec(fft_output);
    del_aubio_mfcc(mfcc);
    del_fvec(input);

    return mfcc_features;
}

int export_mfccFile(std::vector<float> mfcc_features, const char* txtFile){

    std::ofstream FILE(txtFile);

    if(!FILE){
        std::cerr << "Error: Could not open the file for writing\n";
        return 0; 
    }

    for(const float& value : mfcc_features){
        FILE << value << "\n";
    }

    FILE.close();
    std::cout << "MFCCs Features written to " << txtFile << "\n";

    return 0; 
}