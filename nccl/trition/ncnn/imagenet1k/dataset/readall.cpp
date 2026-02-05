#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#include <algorithm>

std::string get_filename(const std::string& full_path) {
    // Find the last occurrence of either '/' or '\'
    size_t last_slash_pos = full_path.find_last_of("/\\");

    // If a slash was found, return the substring after it
    if (last_slash_pos != std::string::npos) {
        return full_path.substr(last_slash_pos + 1);
    }

    // Otherwise, the input string is just a filename
    return full_path;
}
int main() {
    // Define the path and file pattern (e.g., all .jpg files in the 'images' folder)
    String pattern = "/pytorch/ncnn/build/imagenet-sample-images/*.JPEG"; 
    
    // Vector to store the list of filenames
    vector<String> filenames;

    // Use cv::glob to find all files matching the pattern
    glob(pattern, filenames, false); // 'false' for non-recursive search

    // Vector to store the loaded images
    vector<Mat> images;
    size_t count = filenames.size(); // Number of files found

    if (count == 0) {
        cout << "No images found in the directory!" << endl;
        return -1;
    }
    cout << "total jpeg: "  << count << endl;
    // Loop through the filenames and load each image
    for (size_t i = 0; i < count; i++) {
        Mat img = imread(filenames[i]); // Read the image

        // Error handling: check if the image loaded successfully
        if (img.empty()) { 
            cout << "Error: Could not read image " << filenames[i] << endl;
            continue; // Skip to the next iteration
        }
        
        cout << "jpeg: "  << get_filename(filenames[i])<< endl;
	
        //images.push_back(img); // Add the image to the vector

        // Optional: Display each image as it's loaded
        // imshow("Loaded Image", img);
        // waitKey(100); // Display for 100ms or until a key is pressed
    }

    // Example of accessing the first loaded image
    if (!images.empty()) {
        //imshow("First Image", images[0]);
        //waitKey(0); // Wait for a key press to close all windows
    }

    return 0;
}

