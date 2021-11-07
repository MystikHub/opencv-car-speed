#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

#define FOCAL_LENGTH_ESTIMATE 1770
#define PLATE_WIDTH_IN_MM 465
#define PLATE_HEIGHT_IN_MM 100
#define FRAMES_PER_SECOND 29.97
#define REQUIRED_DICE 0.8
#define MAX_PREDICTION_DELTA 100
#define SUPPRESS_FRAME_INFO
const int LICENCE_PLATE_LOCATIONS[][5] = { {1, 67, 88, 26, 6}, {2, 67, 88, 26, 6}, {3, 68, 88, 26, 6},
	{4, 69, 88, 26, 6}, {5, 70, 89, 26, 6}, {6, 70, 89, 27, 6}, {7, 71, 89, 27, 6}, {8, 73, 89, 27, 6},
	{9, 73, 90, 27, 6}, {10, 74, 90, 27, 6}, {11, 75, 90, 27, 6}, {12, 76, 90, 27, 6}, {13, 77, 91, 27, 6},
	{14, 78, 91, 27, 6}, {15, 78, 91, 27, 6}, {16, 79, 91, 27, 6}, {17, 80, 92, 27, 6}, {18, 81, 92, 27, 6},
	{19, 81, 92, 28, 6}, {20, 82, 93, 28, 6}, {21, 83, 93, 28, 6}, {22, 83, 93, 28, 6}, {23, 84, 93, 28, 6},
	{24, 85, 94, 28, 6}, {25, 85, 94, 28, 6}, {26, 86, 94, 28, 6}, {27, 86, 94, 28, 6}, {28, 86, 95, 29, 6},
	{29, 87, 95, 29, 6}, {30, 87, 95, 29, 6}, {31, 88, 95, 29, 6}, {32, 88, 96, 29, 6}, {33, 89, 96, 29, 6},
	{34, 89, 96, 29, 6}, {35, 89, 97, 29, 6}, {36, 90, 97, 29, 6}, {37, 90, 97, 30, 6}, {38, 91, 98, 30, 6},
	{39, 91, 98, 30, 6}, {40, 92, 98, 30, 7}, {41, 92, 99, 30, 7}, {42, 93, 99, 30, 7}, {43, 93, 99, 30, 7},
	{44, 94, 100, 30, 7}, {45, 95, 100, 30, 7}, {46, 95, 101, 30, 7}, {47, 96, 101, 30, 7}, {48, 97, 102, 30, 7},
	{49, 97, 102, 31, 7}, {50, 98, 102, 31, 7}, {51, 99, 103, 31, 7}, {52, 99, 103, 32, 7}, {53, 100, 104, 32, 7},
	{54, 101, 104, 32, 7}, {55, 102, 105, 32, 7}, {56, 103, 105, 32, 7}, {57, 104, 106, 32, 7}, {58, 105, 106, 32, 7},
	{59, 106, 107, 32, 7}, {60, 107, 107, 32, 7}, {61, 108, 108, 32, 7}, {62, 109, 108, 33, 7}, {63, 110, 109, 33, 7},
	{64, 111, 109, 33, 7}, {65, 112, 110, 34, 7}, {66, 113, 111, 34, 7}, {67, 114, 111, 34, 7}, {68, 116, 112, 34, 7},
	{69, 117, 112, 34, 8}, {70, 118, 113, 35, 8}, {71, 119, 113, 35, 8}, {72, 121, 114, 35, 8}, {73, 122, 114, 35, 8},
	{74, 124, 115, 35, 8}, {75, 125, 116, 36, 8}, {76, 127, 116, 36, 8}, {77, 128, 117, 36, 8}, {78, 130, 118, 36, 8},
	{79, 132, 118, 36, 9}, {80, 133, 119, 37, 9}, {81, 135, 120, 37, 9}, {82, 137, 121, 37, 9}, {83, 138, 122, 38, 9},
	{84, 140, 122, 38, 9}, {85, 142, 123, 38, 9}, {86, 144, 124, 38, 9}, {87, 146, 125, 38, 9}, {88, 148, 126, 39, 9},
	{89, 150, 127, 39, 9}, {90, 152, 128, 39, 9}, {91, 154, 129, 40, 9}, {92, 156, 129, 40, 10}, {93, 158, 130, 40, 10},
	{94, 160, 131, 41, 10}, {95, 163, 133, 41, 10}, {96, 165, 133, 41, 10}, {97, 167, 135, 42, 10}, {98, 170, 135, 42, 10},
	{99, 172, 137, 43, 10}, {100, 175, 138, 43, 10}, {101, 178, 139, 43, 10}, {102, 180, 140, 44, 10}, {103, 183, 141, 44, 10},
	{104, 186, 142, 44, 11}, {105, 188, 143, 45, 11}, {106, 192, 145, 45, 11}, {107, 195, 146, 45, 11}, {108, 198, 147, 45, 11},
	{109, 201, 149, 46, 11}, {110, 204, 150, 47, 11}, {111, 207, 151, 47, 11}, {112, 211, 152, 47, 11}, {113, 214, 154, 48, 11},
	{114, 218, 155, 48, 12}, {115, 221, 157, 49, 12}, {116, 225, 158, 50, 12}, {117, 229, 160, 50, 12}, {118, 234, 162, 50, 12},
	{119, 237, 163, 51, 12}, {120, 241, 164, 52, 12}, {121, 245, 166, 52, 12}, {122, 250, 168, 52, 12}, {123, 254, 169, 53, 12},
	{124, 258, 171, 54, 12}, {125, 263, 173, 55, 12}, {126, 268, 175, 55, 12}, {127, 273, 177, 55, 12}, {128, 278, 179, 56, 13},
	{129, 283, 181, 57, 13}, {130, 288, 183, 57, 13}, {131, 294, 185, 58, 13}, {132, 299, 187, 59, 13}, {133, 305, 190, 59, 13},
	{134, 311, 192, 60, 13}, {135, 317, 194, 60, 14}, {136, 324, 196, 60, 14}, {137, 330, 198, 61, 14}, {138, 336, 201, 63, 14},
	{139, 342, 203, 64, 14}, {140, 349, 206, 65, 14}, {141, 357, 208, 65, 15}, {142, 364, 211, 66, 15}, {143, 372, 214, 67, 15},
	{144, 379, 217, 68, 15}, {145, 387, 220, 69, 15}, {146, 396, 223, 70, 15}, {147, 404, 226, 71, 16}, {148, 412, 229, 72, 16},
	{149, 422, 232, 73, 17}, {150, 432, 236, 74, 17}, {151, 440, 239, 75, 18}, {152, 450, 243, 76, 18}, {153, 460, 247, 77, 18},
	{154, 470, 250, 78, 19}, {155, 482, 254, 78, 19}, {156, 492, 259, 81, 19}, {157, 504, 263, 82, 20}, {158, 516, 268, 83, 20},
	{159, 528, 272, 85, 21}, {160, 542, 277, 85, 21}, {161, 554, 282, 88, 21}, {162, 569, 287, 88, 22}, {163, 584, 292, 89, 22},
	{164, 598, 297, 91, 23}, {165, 614, 302, 92, 24}, {166, 630, 308, 94, 24}, {167, 646, 314, 96, 25}, {168, 664, 320, 97, 26},
	{169, 681, 327, 100, 26}, {170, 700, 334, 101, 27}, {171, 719, 341, 103, 28}, {172, 740, 349, 105, 29}, {173, 762, 357, 107, 29},
	{174, 784, 365, 109, 30}, { 175, 808, 374, 110, 31 }, { 176, 832, 383, 113, 32 } };
const int NUMBER_OF_PLATES = sizeof(LICENCE_PLATE_LOCATIONS) / (sizeof(LICENCE_PLATE_LOCATIONS[0]));
const int FRAMES_FOR_DISTANCES[] = { 54,   70,   86,  101,  115,  129,  143,  158,  172 };
const int DISTANCES_TRAVELLED_IN_MM[] = { 2380, 2380, 2400, 2380, 2395, 2380, 2385, 2380 };
const double SPEEDS_IN_KMPH[] = { 16.0, 16.0, 17.3, 18.3, 18.5, 18.3, 17.2, 18.3 };

double distance(Point a, Point b) {
    // Pythagora's theorem, a^2 + b^2 = c^2
    // c, a.k.a. distance = sqrt(x_dist^2 + y_dist^2)
    double x_distance = pow(b.x - a.x, 2);
    double y_distance = pow(b.y - a.y, 2);

    double distance = sqrt(x_distance + y_distance);
    return distance;
}

vector<double> distances;
vector<Rect> estimateBBs;
void printFinalMetrics() {
    // Calculate the distances between each pair of points defined in
    //  FRAMES_FOR_DISTANCES
    double distancesBetween[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for(size_t i = 0; i < distances.size(); i++) {
        // Start processing from distances index 1 to be able to calculate
        //  the difference between this frame and the previous one
        int frameNumber = i + 1;
        int binIndex = -1;

        // Find which bin this frame goes in (index for distancesBetween)
        bool foundBin = false;
        for(int j = 0; j < 8 && !foundBin; j++) {
            if(frameNumber >= FRAMES_FOR_DISTANCES[j] && frameNumber < FRAMES_FOR_DISTANCES[j + 1]) {
                foundBin = true;
                binIndex = j;
            }
        }

        // Current distance data might fall outside the defined ground truth
        //  frames
        if(foundBin) {
            double distanceDelta = distances[frameNumber] - distances[frameNumber - 1];

            // Negative delta because the car is approaching the camera
            distancesBetween[binIndex] += -(distanceDelta); 
        }
    }

    // Print the distances between the ground truth frames and our calculations
    //  (speeds as well!)
    printf("Distances and speeds between frames\n");

    for(int i = 0; i < 8; i++) {
        double differenceFromGroundTruth = abs(distancesBetween[i] - DISTANCES_TRAVELLED_IN_MM[i]);
        printf("%d and %d: %.3fmm (diff of %.3f)", FRAMES_FOR_DISTANCES[i], FRAMES_FOR_DISTANCES[i + 1], distancesBetween[i], differenceFromGroundTruth);

        // Calculate the speed from the distance between each of the ground
        //  truths using the data from the previous loop and the time difference
        //  between each of the 
        double distanceTravelled = distancesBetween[i] / (double) 1000;
        double timeElapsed = ((double) (FRAMES_FOR_DISTANCES[i + 1] - FRAMES_FOR_DISTANCES[i])) / (double) FRAMES_PER_SECOND;

        double speedInMPerSecond = distanceTravelled / timeElapsed;
        double speedInKmPerHour = speedInMPerSecond * (double) 3.6;

        differenceFromGroundTruth = abs(speedInKmPerHour - SPEEDS_IN_KMPH[i]);
        printf("\taverage %.3fkm/h (diff of %.3f)\n", speedInKmPerHour, differenceFromGroundTruth);
    }
    printf("\n");

    // Find DICE coefficient metrics;
    int truePositives = 0;
    int falsePositives = 0; // Frames where there wasn't a license plate and
                            //  we predicted there was a license plate
    int falseNegatives = 0; // Frames where there was a license plate and we 
                            //  didn't have a good enough dice coefficient
    for(size_t i = 0; i < distances.size(); i++) {
        // Ground truth rectangle
        const int* gtDimensions = LICENCE_PLATE_LOCATIONS[i];
        Rect groundTruthRectangle = Rect(gtDimensions[1], gtDimensions[2], gtDimensions[3], gtDimensions[4]);
        Rect intersectionRectangle = estimateBBs[i] & groundTruthRectangle;

        double intersectionArea = intersectionRectangle.area();
        double sumOfAreas = groundTruthRectangle.area() + estimateBBs[i].area();

        double diceCoefficient = ((double) 2 * intersectionArea) / sumOfAreas;

        // If we have ground truth data for these frames
        if(i < NUMBER_OF_PLATES) {
            if(diceCoefficient >= 0.8) {
                truePositives++;
            } else {
                falseNegatives++;
            }
        // We made predictions, but there was no ground truth
        } else {
            // My system always has predictions, even for frames without a
            //  license plate, so all of those predictions are false positives
            falsePositives++;
        }
    }

    printf("True positives: %d\n", truePositives);
    printf("False positives: %d\n", falsePositives);
    printf("False negatives: %d\n\n", falseNegatives);

    // Calculate precision and recall
    double precision = (double) (truePositives) / ((double) truePositives + falsePositives);
    double recall = (double) (truePositives) / (double) (truePositives + falseNegatives);
    printf("Precision: %.3f%%\n", precision * 100);
    printf("Recall: %.3f%%\n", recall * 100);

    // Clear our vector of saved distances for the next loop through the video
    distances.clear();
    estimateBBs.clear();
}

int main(int argc, char** argv) {
    // Load the video
    VideoCapture carVideo("data/CarSpeedTest1.mp4");
    Mat staticBackground = imread("data/CarSpeedTest1EmptyFrame.jpg", 1);
    printf("carVideo.isOpened: %d\n", carVideo.isOpened());
    
    // Set up windows showing each step
    namedWindow("Car video");
    namedWindow("Background removed and thresholded");
    namedWindow("Moving points");
    namedWindow("Masked and thresholded moving points");
    namedWindow("License plate location");

    // Data used between processing each frame
    bool videoFinished = false;
    int frameNumber = 1;
    Rect previouslyFoundLocation = Rect(0, 0, 0, 0);
    double previousDistance = 0, previousSpeed = 0;

    // Load the first frame (but do nothing with it, ground truth is offset by one)
    Mat current_frame;
    carVideo >> current_frame;

    // Process each frame of the video until the user presses ESC
    while(!videoFinished) {
        carVideo >> current_frame;

        // Wait for this frame to appear for 1/29.97 seconds and read user input
        char buttonPressed = (char) waitKey((double) 1000 / (double) FRAMES_PER_SECOND);

        // Stop if the user pressed ESC
        if(buttonPressed == 27) {
            videoFinished = true;

        // User can pause the video with space (32) then continue or stop the playback
        } else if(buttonPressed == 32 && waitKey(0) == 27) {
            videoFinished = true;

        // No more video data, restart the playback and print metrics
        } else if(current_frame.empty()) {
            carVideo = VideoCapture("data/CarSpeedTest1.mp4");
            carVideo >> current_frame;
            frameNumber = 1;

            printFinalMetrics();

        // Process the video normally
        } else {

            // Ground truth rectangle
            const int* thisLocationData = LICENCE_PLATE_LOCATIONS[frameNumber - 1];
            Rect rect(thisLocationData[1], thisLocationData[2], thisLocationData[3], thisLocationData[4]);
            rectangle(current_frame, rect, Scalar(0, 255, 0));
            imshow("Car video", current_frame);

            // Use the static background to mask and threshold different pixels
            // This should leave a lot of pixels from the car
            Mat noBackground, grayscaleNoBackground, movingPoints;
            absdiff(current_frame, staticBackground, noBackground);
            cvtColor(noBackground, grayscaleNoBackground, COLOR_BGR2GRAY);
            threshold(grayscaleNoBackground, movingPoints, 15, 255, THRESH_BINARY);
            imshow("Background removed and thresholded", movingPoints);

            // Dilate the thresholded mask from the static background and filter
            //  out the moving pixels from the video
            Mat dilatedMovingPoints, newThreshold;
            Mat structuringElement = getStructuringElement(MORPH_RECT, Size(3, 3));
            dilate(movingPoints, dilatedMovingPoints, structuringElement);
            dilate(movingPoints, dilatedMovingPoints, structuringElement);
            current_frame.copyTo(newThreshold, dilatedMovingPoints);
            imshow("Moving points", newThreshold);

            // Do an aggressive threshold on the moving pixels, then erode away
            //  some noise and dilate twice to get a wider outline of the
            //  license plate
            Mat licensePlate, erodedLicensePlate, dilatedLicensePlate, unlocatedLicensePlate;
            cvtColor(newThreshold, newThreshold, COLOR_BGR2GRAY);
            threshold(newThreshold, licensePlate, 50, 255, THRESH_BINARY);
            erode(licensePlate, erodedLicensePlate, structuringElement);
            dilate(erodedLicensePlate, dilatedLicensePlate, structuringElement);
            dilate(dilatedLicensePlate, unlocatedLicensePlate, structuringElement);
            rectangle(unlocatedLicensePlate, rect, Scalar(0, 0, 255));
            imshow("Masked and thresholded moving points", unlocatedLicensePlate);

            // Find the license plate
            vector<vector<Point>> regions;
            Mat licensePlateImage = current_frame;
            Rect licensePlateBoundingRectangle;
            findContours(unlocatedLicensePlate, regions, RETR_LIST, CHAIN_APPROX_NONE);

            // Find the region with the highest rectangularity
            int indexOfContourWithLargestRectangularity = -1;
            double sizeOfLargestRectangularity = 0;

            for(size_t i = 0; i < regions.size(); i++) {

                // Find the minimum bounding rectangle around this region
                Rect boundingRectangle = boundingRect(regions[i]);
                double rectangularity = contourArea(regions[i]) / ((boundingRectangle.width) * (boundingRectangle.height));

                // Save this region's index if it has the highest rectangularity
                if(rectangularity > sizeOfLargestRectangularity
                        && boundingRectangle.width > boundingRectangle.height) {
                        // && distance(previouslyFoundLocation.tl(), boundingRectangle.tl()) < MAX_PREDICTION_DELTA) {

                    sizeOfLargestRectangularity = rectangularity;
                    indexOfContourWithLargestRectangularity = i;
                }
            }

            // Save this license plate location and draw a rectangle of it
            //  the check on regions.size() prevents a crash at the end of the
            //  video
            if(regions.size() && indexOfContourWithLargestRectangularity != -1) {
                // Save this region's bounding rectangle for distance calculation
                licensePlateBoundingRectangle = boundingRect(regions[indexOfContourWithLargestRectangularity]);

                // And draw it on our window
                rectangle(licensePlateImage, licensePlateBoundingRectangle, Scalar(0, 0, 255));

                // Unused variable from restricting the distance between the
                //  license plate between frames
                previouslyFoundLocation = Rect(licensePlateBoundingRectangle);
            }
            imshow("License plate location", licensePlateImage);
            
            // Calculate the distance to the object
            // For the width and height of our estimate of the license plate shape:
            //
            // screen width/height (pixels) / focal length (pixels) = real width/height (mm) / distance to license plate (x/y, mm)
            //
            // i.e. distance to license plate(x/y, mm) = real width/height (mm) * focal length (pixels) / screen width/height (pixels)


            // Rescale the bounding box to match the license plate
            //  omitted due to poorer results
            // double licensePlateAspectRatio = (double) PLATE_WIDTH_IN_MM / (double) PLATE_HEIGHT_IN_MM;
            // double estimateAspectRatio = (double) licensePlateBoundingRectangle.width / (double) licensePlateBoundingRectangle.height;
            // licensePlateBoundingRectangle.width *= licensePlateAspectRatio / estimateAspectRatio;

            // Calculate the distance using the formula above
            double xDistance = ((double) PLATE_WIDTH_IN_MM * (double) FOCAL_LENGTH_ESTIMATE) / (double) licensePlateBoundingRectangle.width;
            double yDistance = ((double) PLATE_HEIGHT_IN_MM * (double) FOCAL_LENGTH_ESTIMATE) / (double) licensePlateBoundingRectangle.height;
            double fullDistance = distance(Point(0, 0), Point(xDistance, yDistance)) / 1000;    // Divide by 1000 to convert millimeters to meters

            // Calculate the speed
            // speed = distance / time
            double distanceDelta = fullDistance - previousDistance;
            double speed = distanceDelta / ((double) 1 / (double) FRAMES_PER_SECOND);

            // Sometimes, the distance calculation between frames might result
            // in the same distance, so use the previous speed to come up with a
            // non-zero answer
            previousDistance = fullDistance;

            if(speed == 0)
                speed = previousSpeed;
            else
                previousSpeed = speed;

            // Save this image's results for metrics
            distances.push_back(fullDistance * 1000);
            estimateBBs.push_back(licensePlateBoundingRectangle);

            // Print the frame number, distance, and speed to stdout
            double printableSpeed = abs(speed * 3.6);
#ifndef SUPPRESS_FRAME_INFO
            printf("Frame number: %d, distance: %.3fm, speed: %.3f km/h\n", frameNumber, fullDistance, printableSpeed);
#endif
            frameNumber++;
        }
    }

    return 0;
}