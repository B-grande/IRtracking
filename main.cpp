#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "tracker_c.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

// Global detection parameters (with trackbars in the Camera window)
int threshold_value = 200; // Brightness threshold
int min_area = 5;          // Minimum blob area (pixels)
int max_area = 100;        // Maximum blob area (pixels)

// Toggles for additional features (still available if needed)
bool faceTrackingEnabled = true;
bool showBlackWhite = false;

// --- Global variables for multithreaded blob detection --- //
mutex blobMutex;
condition_variable blobCV;
bool newFrameAvailable = false;
Mat sharedGrayFrame;               // The most recent grayscale frame
vector<Blob> sharedBlobs;          // Blob detection results (updated by blob thread)
int blobCapacity;                  // Reserved capacity chosen at startup
bool stopBlobThread = false;       // Flag to signal blob thread to stop

// Blob detection thread function with morphological pre‐processing
void blobDetectionThreadFunction() {
    while (true) {
        unique_lock<mutex> lock(blobMutex);
        // Wait until a new frame is available or stop signal is set.
        blobCV.wait(lock, [] { return newFrameAvailable || stopBlobThread; });
        if (stopBlobThread) break;
        
        // Make a local copy of the shared grayscale frame.
        Mat localGray = sharedGrayFrame.clone();
        lock.unlock();
        
        // --- Pre-processing to reduce background noise ---
        // Apply a binary threshold to isolate bright IR regions.
        Mat thresholded;
        threshold(localGray, thresholded, threshold_value, 255, THRESH_BINARY);
        
        // Create a small rectangular kernel.
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        
        // Apply morphological opening (erosion followed by dilation) to remove small noise.
        Mat filtered;
        morphologyEx(thresholded, filtered, MORPH_OPEN, kernel);
        // --- End of pre‐processing ---
        
        // Allocate a local vector with the reserved capacity.
        vector<Blob> localBlobs(blobCapacity);
        // Use the filtered image for blob detection.
        int num_blobs = detect_blobs(filtered.data, filtered.cols, filtered.rows, filtered.step,
                                      threshold_value, min_area, max_area,
                                      localBlobs.data(), blobCapacity);
        localBlobs.resize(num_blobs);
        
        // Update the shared blobs with the results.
        lock.lock();
        sharedBlobs = localBlobs;
        newFrameAvailable = false;
        lock.unlock();
    }
}

// Function to show a startup window ("Memory Setup") that lets you choose the reserved capacity.
int getReservedCapacityFromUser() {
    int capacity = 50; // Default value
    const int maxCapacity = 1000;
    namedWindow("Memory Setup", WINDOW_AUTOSIZE);
    createTrackbar("Reserved Capacity", "Memory Setup", &capacity, maxCapacity);
    cout << "Adjust the 'Reserved Capacity' trackbar in the 'Memory Setup' window and press 'S' to start." << endl;
    while (true) {
        Mat dummy = Mat::zeros(100, 400, CV_8UC3);
        string text = "Reserved Capacity: " + to_string(capacity);
        putText(dummy, text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        imshow("Memory Setup", dummy);
        char key = (char)waitKey(30);
        if (key == 's' || key == 'S') break;
    }
    destroyWindow("Memory Setup");
    return capacity;
}

int main() {
    // --- Get reserved capacity from user at startup --- //
    blobCapacity = getReservedCapacityFromUser();
    
    // Open the default camera.
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }
    
    // Load the face cascade classifier.
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Error: Could not load face cascade classifier. Ensure haarcascade_frontalface_default.xml is in the working directory." << endl;
        return -1;
    }
    
    // Create windows for Camera and Tracker.
    namedWindow("Camera", WINDOW_AUTOSIZE);
    namedWindow("Tracker", WINDOW_AUTOSIZE);
    
    // Create trackbars for detection parameters in the "Camera" window.
    createTrackbar("Threshold", "Camera", &threshold_value, 255);
    createTrackbar("Min Area", "Camera", &min_area, 500);
    createTrackbar("Max Area", "Camera", &max_area, 1000);
    
    // Start the blob detection thread.
    thread blobThread(blobDetectionThreadFunction);
    
    // For overlay timing (when detection parameters change).
    int prevMinArea = min_area;
    int prevMaxArea = max_area;
    steady_clock::time_point areaChangeTime = steady_clock::now();
    
    // Variables for gesture control.
    // We'll track the average center of detected blobs.
    Point prevGestureCenter(0, 0);
    bool firstGesture = true;
    auto lastGestureTime = steady_clock::now();
    const int gestureThreshold = 50; // pixels
    const int gestureCooldownMs = 1000; // 1 second cooldown
    
    // For displaying the command
    string gestureCommand = "";
    steady_clock::time_point lastCommandTime = steady_clock::now() - milliseconds(2000);
    const int commandDisplayDurationMs = 2000; // Display for 2 seconds
    
    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Convert the frame to grayscale.
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Update overlay timer if min/max area values changed.
        if (min_area != prevMinArea || max_area != prevMaxArea) {
            areaChangeTime = steady_clock::now();
            prevMinArea = min_area;
            prevMaxArea = max_area;
        }
        auto elapsed = duration_cast<seconds>(steady_clock::now() - areaChangeTime);
        
        // --- Update the shared grayscale frame for blob detection --- //
        {
            lock_guard<mutex> lock(blobMutex);
            sharedGrayFrame = gray.clone();
            newFrameAvailable = true;
        }
        blobCV.notify_one();
        
        // --- Face Tracking --- //
        if (faceTrackingEnabled) {
            vector<Rect> faces;
            faceCascade.detectMultiScale(gray, faces);
            for (const auto &face : faces) {
                rectangle(frame, face, Scalar(255, 0, 0), 2);
                int centerX = face.x + face.width / 2;
                int centerY = face.y + face.height / 2;
                circle(frame, Point(centerX, centerY), 3, Scalar(255, 0, 0), -1);
            }
        }
        
        // --- Draw Blob Detection Results (from sharedBlobs) and compute gesture center --- //
        Point gestureCenter(0, 0);
        int count = 0;
        {
            lock_guard<mutex> lock(blobMutex);
            for (const auto &blob : sharedBlobs) {
                circle(frame, Point(blob.x, blob.y), 5, Scalar(0, 0, 255), 2);
                string text = "(" + to_string(blob.x) + "," + to_string(blob.y) + ")";
                putText(frame, text, Point(blob.x + 10, blob.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                gestureCenter.x += blob.x;
                gestureCenter.y += blob.y;
                count++;
            }
        }
        if (count > 0) {
            gestureCenter.x /= count;
            gestureCenter.y /= count;
        }
        
        // --- Gesture Control --- //
        auto now = steady_clock::now();
        if (count > 0 && !firstGesture && duration_cast<milliseconds>(now - lastGestureTime).count() > gestureCooldownMs) {
            int diffX = gestureCenter.x - prevGestureCenter.x;
            if (diffX < -gestureThreshold) {
                // Left swipe detected: set command text.
                gestureCommand = "Left Swipe Command";
                lastCommandTime = now;
                lastGestureTime = now;
            } else if (diffX > gestureThreshold) {
                // Right swipe detected: set command text.
                gestureCommand = "Right Swipe Command";
                lastCommandTime = now;
                lastGestureTime = now;
            }
        }
        if (count > 0) {
            prevGestureCenter = gestureCenter;
            firstGesture = false;
            // Optionally, visualize the gesture center.
            circle(frame, gestureCenter, 8, Scalar(0, 255, 255), -1);
        }
        
        // Display gesture command if within the display duration.
        if (duration_cast<milliseconds>(now - lastCommandTime).count() < commandDisplayDurationMs) {
            putText(frame, gestureCommand, Point(10, frame.rows - 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        }
        
        // --- Optionally overlay detection parameter values for 2 seconds after change --- //
        if (elapsed.count() < 2) {
            string areaText = "Min Area: " + to_string(min_area) + "  Max Area: " + to_string(max_area);
            putText(frame, areaText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        }
        
        // --- Build the Tracker Window (Mapping blobs to a coordinate plane) --- //
        Mat tracker_img = Mat::zeros(500, 500, CV_8UC3);
        for (int i = 50; i < 500; i += 50) {
            line(tracker_img, Point(i, 0), Point(i, 500), Scalar(50, 50, 50), 1);
            line(tracker_img, Point(0, i), Point(500, i), Scalar(50, 50, 50), 1);
        }
        {
            lock_guard<mutex> lock(blobMutex);
            for (const auto &blob : sharedBlobs) {
                int x_coord = (blob.x * tracker_img.cols) / gray.cols;
                int y_coord = (blob.y * tracker_img.rows) / gray.rows;
                circle(tracker_img, Point(x_coord, y_coord), 5, Scalar(0, 255, 0), -1);
            }
        }
        
        // --- Display the Camera Window (optionally in black & white) --- //
        if (showBlackWhite) {
            Mat bwFrame;
            cvtColor(frame, bwFrame, COLOR_BGR2GRAY);
            imshow("Camera", bwFrame);
        } else {
            imshow("Camera", frame);
        }
        imshow("Tracker", tracker_img);
        
        // --- Key Controls --- //
        char key = (char)waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == 'f' || key == 'F')
            faceTrackingEnabled = !faceTrackingEnabled;
        if (key == 'b' || key == 'B')
            showBlackWhite = !showBlackWhite;
    }
    
    // Signal the blob detection thread to stop and wait for it.
    {
        lock_guard<mutex> lock(blobMutex);
        stopBlobThread = true;
        newFrameAvailable = true;
    }
    blobCV.notify_one();
    blobThread.join();
    
    cap.release();
    destroyAllWindows();
    return 0;
}