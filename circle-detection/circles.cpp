#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

int min_circle_dist = 100;
int canny_thresh = 190;
int accumulator_thresh = 24;
int min_radius = 60;
int max_radius = 200;

cv::Mat src, src_gray, image, display;
const char* windowName = "Testing Window";
void LoadImage(const char* fileName, cv::Mat &src, cv::Mat &src_gray)
{
    src = cv::imread(fileName, 1);
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
}
void SmoothImage(cv::Mat &src, cv::Mat &dst)
{
    //do a closing morphology with cross kernel of size 2*9 + 1
    int m_size = 5;
    cv::GaussianBlur(src, dst, cv::Size(11, 11), 2, 2);
    cv::Mat element = cv::getStructuringElement(2, cv::Size(7, 7));
    cv::morphologyEx(src, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 0, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    //cv::morphologyEx(dst, dst, 0, element);
    //
}
void DisplayImage(const char* window_name, cv::Mat &src, cv::Size dimensions)
{
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, dimensions);
    cv::imshow(window_name, src);
    cv::waitKey(0);
}
void DrawCircles(cv::Mat &src, cv::Mat &dst, std::vector<cv::Vec3f> &circles)
{
    cv::HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 65, 50, 70, 70, 200);
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(dst, center, radius, cv::Scalar(0, 0, 255), 3);
    }
}
//code for testing paramters
void Hough_Operations(int, void*)
{
    std::vector<cv::Vec3f> circles;
    display = image.clone();
    cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 1, min_circle_dist, 
                     canny_thresh, accumulator_thresh, min_radius, max_radius);
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(display, center, radius, cv::Scalar(0, 0, 255), 3);
    }
    cv::imshow(windowName, display);
}
void TestCircles()
{
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1600, 1200);
    cv::createTrackbar("Minimum Circle Distance", windowName, &min_circle_dist, 100, Hough_Operations);
    cv::createTrackbar("Canny Threshhold", windowName, &canny_thresh, 200, Hough_Operations);
    cv::createTrackbar("Accumulator Threshhold", windowName, &accumulator_thresh, 200, Hough_Operations);
    Hough_Operations(0, 0);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{ 
    //cv::Mat src, src_gray, image;
    LoadImage(argv[1], src, src_gray); //load image
    SmoothImage(src_gray, image); //apply morphology
    TestCircles();
    //DisplayImage("Smoothed Image", image, cv::Size(1600, 1200)); //testing purposes
    //apply circle detection
    std::vector<cv::Vec3f> circles;
    DrawCircles(image, image, circles);
    //DisplayImage(windowName, image, cv::Size(1600, 1200));

    // std::vector<cv::Mat> images;
    // images.push_back(src_gray);
    // cv::MatND hist;
    // int histSize = 256;
    // float range[] = { 0, 256 }; //the upper boundary is exclusive
    // const float* histRange = { range };
    // cv::calcHist(&src_gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    // //int hist_w = 512, hist_h = 400;
    // //int bin_w = cvRound((double)hist_w / histSize);
    // //cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    // cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    // for(int i = 0; i < src.rows; i++)
    // {
    //     for(int j = 0; j < src.cols; j++)
    //     {
    //         int grayscale_intensity = src_gray.at<uchar>(i, j);
    //         double hist_val = hist.at<float>(grayscale_intensity);
    //         if(grayscale_intensity > 140 && hist_val > 0.25)
    //             src.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
    //         else
    //             src.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
    //     }
    // }
    // //display the image
    // cv::namedWindow("image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("image", 1600, 1200);
    // cv::imshow("image", src);
    // cv::waitKey(0);
}