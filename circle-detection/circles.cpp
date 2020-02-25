#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

int main(int argc, char** argv)
{
    cv::Mat src, src_gray;

    //load the image to color and grayscale
    src = cv::imread(argv[1], 1);
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    //apply gaussian blur
    cv::GaussianBlur(src_gray, src_gray, cv::Size(7, 7), 2, 2);
    //apply circle detection
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(src_gray, circles, cv::HOUGH_GRADIENT, 1, 65, 65, 50, 70, 200);

    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(src, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
    }
    std::vector<cv::Mat> images;
    images.push_back(src_gray);
    cv::MatND hist;
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    cv::calcHist(&src_gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    //int hist_w = 512, hist_h = 400;
    //int bin_w = cvRound((double)hist_w / histSize);
    //cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            int grayscale_intensity = src_gray.at<uchar>(i, j);
            double hist_val = hist.at<float>(grayscale_intensity);
            if(grayscale_intensity > 140 && hist_val > 0.25)
                src.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            else
                src.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
        }
    }
    //display the image
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::resizeWindow("image", 1600, 1200);
    cv::imshow("image", src);
    cv::waitKey(0);
}