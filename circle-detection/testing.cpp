#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
//*****Global Variables & Constants*****
int min_circle_dist = 100;
int canny_thresh = 180;
int accumulator_thresh = 24;
int min_radius = 60;
int max_radius = 200;
double bg_intensity = 0.4;
double bg_ratio = 0.11;
const char* windowName = "Testing Window";
//*****Image I/O Functions*****
void LoadImage(const char* fileName, cv::Mat &src, cv::Mat &src_gray)
{
    src = cv::imread(fileName, 1);
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
}
void DisplayImage(const char* window_name, cv::Mat &src, cv::Size dimensions)
{
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, dimensions);
    cv::imshow(window_name, src);
    cv::waitKey(0);
}
//*****Image Processing Functions*****
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
}
void MaskBackground(cv::Mat &src, cv::Mat &dst)
{
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    std::vector<cv::Mat> hist_channels(3);
    for(int i = 0; i < 3; i++)
    {
        calcHist(&channels[i], 1, 0, cv::Mat(), hist_channels[i], 1, &histSize, &histRange);
        normalize(hist_channels[i], hist_channels[i], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    } 
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            double b_freq = hist_channels[0].at<float>(pixel[0]);
            double g_freq = hist_channels[1].at<float>(pixel[1]);
            double r_freq = hist_channels[2].at<float>(pixel[2]);
            //printf("%f %f %f\n", b_freq, g_freq, r_freq);
            if(b_freq > bg_intensity && g_freq > bg_intensity && r_freq > bg_intensity)
                dst.at<uchar>(i, j) = 1;
            else
                dst.at<uchar>(i, j) = 0;
        }
    }
}
void CleanCircles(std::vector<cv::Vec3f> &circles, cv::Mat &bg_mask)
{
    std::vector<cv::Vec3f> new_circles;
    cv::Mat coin_mask, intersection;
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        coin_mask = cv::Mat::zeros(bg_mask.size(), CV_8UC1);
        cv::circle(coin_mask, center, radius - 5, 1, -1);
        int pixels_in_coin = cv::countNonZero(coin_mask);
        cv::bitwise_and(bg_mask, coin_mask, intersection);
        int bg_pixels = cv::countNonZero(intersection);
        if(bg_pixels <= pixels_in_coin * bg_ratio)
            new_circles.push_back(circles[i]);
    }
    circles = new_circles;
}
void ClassifyCoins(cv::Mat &src, std::vector<cv::Vec3f> &circles, std::vector<cv::Vec3f> &pennies, std::vector<cv::Vec3f> &nonpennies)
{
    cv::Mat src_hsv;
    cv::cvtColor(src, src_hsv, cv::COLOR_BGR2HSV);
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::Mat coin_mask = cv::Mat::zeros(src.size(), CV_8UC1);
        cv::circle(coin_mask, center, radius - 5, 1, -1);
        cv::Scalar avg = cv::mean(src, coin_mask);
        if(avg[0] < 100)
            pennies.push_back(circles[i]);
        else
            nonpennies.push_back(circles[i]);
    }
}
void DrawCircles(cv::Mat &src, std::vector<cv::Vec3f> &pennies, std::vector<cv::Vec3f> &nonpennies)
{
    for(int i = 0; i < pennies.size(); i++)
    {
        cv::Point center(cvRound(pennies[i][0]), cvRound(pennies[i][1]));
        int radius = cvRound(pennies[i][2]);
        cv::circle(src, center, radius, cv::Scalar(255, 0, 0), 3);
    }
    for(int i = 0; i < nonpennies.size(); i++)
    {
        cv::Point center(cvRound(nonpennies[i][0]), cvRound(nonpennies[i][1]));
        int radius = cvRound(nonpennies[i][2]);
        cv::circle(src, center, radius, cv::Scalar(0, 0, 255), 3);
    }
}

int main(int argc, char** argv)
{ 
    cv::Mat src, src_gray, image;
    LoadImage(argv[1], src, src_gray); //load image
    
    cv::Rect roi(100, 100, 200, 200);
    cv::Mat imageroi = src(roi);
    cv::circle(imageroi, cv::Point(0, 0), 30, cv::Scalar(255, 0, 0));
    DisplayImage("roi", imageroi, cv::Size(1600, 1200));
    cv::waitKey();
    DisplayImage("not roi", src, cv::Size(1600, 1200));
    cv::waitKey();
}