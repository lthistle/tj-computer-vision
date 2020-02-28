#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
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
int GetPixelLocation(cv::Mat &src, int y, int x)
{
    return src.cols * y + x;
}
//flood fill to find connected components,
//then remove components less than a certain size
void RemoveCoinsFromMask(cv::Mat &src, cv::Mat &mask)
{
    //run connected components on the background
    cv::Mat labelImage, stats, centroids;
    std::unordered_set<int> largeEnoughLabels;
    int nLabels = cv::connectedComponentsWithStats(mask, labelImage, stats, centroids, 4);
    //find all labels that are large enough (meaning not noise in coin)
    for(int i = 0; i < nLabels; i++)
        if(stats.at<int>(i, cv::CC_STAT_AREA) > 20000)
            largeEnoughLabels.insert(i);
    for(int i = 0; i < mask.rows; i++)
        for(int j = 0; j < mask.cols; j++)
        {
            int label = labelImage.at<int>(i, j);
            if(largeEnoughLabels.find(label) == largeEnoughLabels.end())
            {
                mask.at<uchar>(i, j) = 0;
            }
                
        }
}
void DisplayImage(const char* window_name, cv::Mat &src, cv::Size dimensions)
{
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, dimensions);
    cv::imshow(window_name, src);
}
int main(int argc, char** argv)
{ 
    cv::Mat src, src_gray, image;
    LoadImage(argv[1], src, src_gray); //load image
    //split the color image into 3 color channels, smooth it, then recombine
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);
    for(int i = 0; i < 3; i++)
        SmoothImage(channels[i], channels[i]);
    cv::merge(channels, image);
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;        
    cv::Mat b_hist, g_hist, r_hist;
    calcHist( &channels[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &channels[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &channels[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              cv::Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              cv::Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              cv::Scalar( 0, 0, 255), 2, 8, 0  );
    }
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::resizeWindow("image", 1200, 1600);
    cv::imshow("image", image);
    //cv::waitKey();
    cv::Mat background_mask = cv::Mat::zeros(image.size(), CV_8UC1);
    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            int b_freq = b_hist.at<float>(pixel[0]);
            int g_freq = g_hist.at<float>(pixel[1]);
            int r_freq = r_hist.at<float>(pixel[2]);
            int t_min = 100;
            if(b_freq > t_min && g_freq > t_min && r_freq > t_min)
                background_mask.at<uchar>(i, j) = 1;
            else
                background_mask.at<uchar>(i, j) = 0;
        }
    }
    cv::Mat display1, display2;
    src.copyTo(display1, 1 - background_mask);
    DisplayImage("v1", display1, cv::Size(1600, 1200));
    RemoveCoinsFromMask(image, background_mask);
    cv::Mat inverse_mask = 1 - background_mask;
    src.copyTo(display2, inverse_mask);
    DisplayImage("v2", display2, cv::Size(1600, 1200));
    cv::waitKey();

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