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

double VIDEO_FRAMERATE = 29.97;

int main(int argc, char** argv) {
    // Make sure a single argument has been given
    if(argc != 2) {
        printf("usage: vehicle-speed.out <path-to-image-directory>\n");
        return -1;
    }

    VideoCapture carVideo("data/CarSpeedTest1.mp4");
    Mat staticBackground = imread("data/CarSpeedTest1EmptyFrame.jpg", 1);
    printf("carVideo.isOpened: %d\n", carVideo.isOpened());
    
    namedWindow("Car video");
    namedWindow("Moving points");
    namedWindow("Masked and thresholded moving points");
    bool videoFinished = false;
    int frameNumber = 1;
    while(!videoFinished) {
        Mat current_frame;
        carVideo >> current_frame;
        if(current_frame.empty()) {
            videoFinished = true;
        } else {
            imshow("Car video", current_frame);
        }

        char buttonPressed = (char) waitKey((double) 1000 / (double) VIDEO_FRAMERATE);
        if(buttonPressed == 27) {
            videoFinished = true;
        } else if(buttonPressed == 32 && waitKey(0) == 27) {
            videoFinished = true;
        }

        // Use the static background to mask different pixels
        // This should leave a lot of pixels from the car
        Mat noBackground, grayscaleNoBackground, movingPoints;
        absdiff(current_frame, staticBackground, noBackground);
        cvtColor(noBackground, grayscaleNoBackground, COLOR_BGR2GRAY);
        threshold(grayscaleNoBackground, movingPoints, 15, 255, THRESH_BINARY);

        // Erode the masked and thresholded image
        Mat erodedMovingPoints, dilatedMovingPoints, newThreshold;
        Mat structuringElement = getStructuringElement(MORPH_RECT, Size(3, 3));
        // erode(movingPoints, erodedMovingPoints, structuringElement);
        dilate(movingPoints, dilatedMovingPoints, structuringElement);
        dilate(movingPoints, dilatedMovingPoints, structuringElement);
        current_frame.copyTo(newThreshold, dilatedMovingPoints);
        imshow("Moving points", newThreshold);

        // Masked and thresholded moving points
        Mat licensePlate, erodedLicensePlate, unlocatedLicensePlate;
        cvtColor(newThreshold, newThreshold, COLOR_BGR2GRAY);
        threshold(newThreshold, licensePlate, 60, 255, THRESH_BINARY);
        erode(licensePlate, erodedLicensePlate, structuringElement);
        dilate(erodedLicensePlate, erodedLicensePlate, structuringElement);
        dilate(erodedLicensePlate, unlocatedLicensePlate, structuringElement);
        imshow("Masked and thresholded moving points", unlocatedLicensePlate);

        // Testing how rectangles can be drawn
        // int x = 100;
        // int y = 100;
        // int width = 200;
        // int height = 100;

        // Rect rect(x, y, width, height);
        // Point pt1(x, y);
        // Point pt2(x + width, y + height);
        // rectangle(current_frame, pt1, pt2, Scalar(0, 0, 255));

        // imshow("With rectangle", current_frame);
        printf("Frame number: %d\n", frameNumber++);
    }

    return 0;
}