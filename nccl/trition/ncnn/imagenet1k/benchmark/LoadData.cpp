#include <iostream>
#include <fstream>
#include <cstdint>
#include <random>
#include "LoadData.h"

using namespace std;

/*uint8_t images[NUM_IMAGES][IMAGE_SIZE];
uint8_t labels[NUM_IMAGES];*/

void loadCIFAR10(const string filePaths[], int numFiles, uint8_t images[NUM_IMAGES][IMAGE_SIZE], uint8_t labels[NUM_IMAGES]) {
    int imageIndex = 0; // To track the current index in the images and labels arrays

    for (int fileIdx = 0; fileIdx < numFiles; fileIdx++) {
        ifstream file(filePaths[fileIdx], ios::binary);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filePaths[fileIdx] << endl;
            continue; // Skip this file and move to the next
        }

        while (imageIndex < NUM_IMAGES && file.peek() != EOF) {
            file.read(reinterpret_cast<char*>(&labels[imageIndex]), sizeof(labels[imageIndex])); // Read label
            file.read(reinterpret_cast<char*>(images[imageIndex]), IMAGE_SIZE);                 // Read image data
            ++imageIndex;
        }

        file.close();
    }

    if (imageIndex < NUM_IMAGES) {
        cerr << "Warning: Not all images were loaded. Expected " << NUM_IMAGES << ", but loaded " << imageIndex << "." << endl;
    }
}

void initializeWeights(float min, float max, int size, float* weights, unsigned int seed){
    //random_device rd;
    default_random_engine eng(seed);
    uniform_real_distribution<float> dist(min, max);

    for(int i = 0; i < size; i++){
        weights[i] = dist(eng);
    }
}