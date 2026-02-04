#ifndef LOADDATA_H
#define LOADDATA_H

#include <vector>
#include <string>
#include <cstdint>

using namespace std;

const int IMAGE_SIZE = 32 * 32 * 3; // 3,072 bytes per image
const int NUM_IMAGES = 50000;      // 10,000 images per training set and 5 training sets

// Arrays for storing images and labels
extern uint8_t images[NUM_IMAGES][IMAGE_SIZE];
extern uint8_t labels[NUM_IMAGES];

// Function to load CIFAR-10 data from a binary file
void loadCIFAR10(const string filePaths[], int numFiles, uint8_t images[NUM_IMAGES][IMAGE_SIZE], uint8_t labels[NUM_IMAGES]);
void initializeWeights(float min, float max, int size, float* weights, unsigned int seed);

#endif