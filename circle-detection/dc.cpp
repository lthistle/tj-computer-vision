//Luke Thistlethwaite
//Coin Detection using OpenCV (4.2.0)
//Period 2
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <unordered_set>
//*****Global Variables & Constants*****
int min_circle_dist = 100;
int canny_thresh = 180;
int accumulator_thresh = 24;
int min_radius = 60;
int max_radius = 200;
double bg_intensity = 0.4;
double bg_ratio = 0.11;
double max_hue = 0;
double min_hue = 256;
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
    cv::Mat element = cv::getStructuringElement(2, cv::Size(9, 9));
    cv::morphologyEx(src, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 0, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
    cv::morphologyEx(dst, dst, 2, element);
}
void GetBackgroundMask(cv::Mat &src, cv::Mat &dst)
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
            if(b_freq > bg_intensity && g_freq > bg_intensity && r_freq > bg_intensity)
                dst.at<uchar>(i, j) = 1;
            else
                dst.at<uchar>(i, j) = 0;
        }
    }
}
void CleanCircles(std::vector<cv::Vec3f> &circles, cv::Mat &src, cv::Mat &bg_mask, std::unordered_map<int, double> &hue_map)
{
    std::vector<cv::Vec3f> new_circles;
    cv::Mat coin_mask, intersection;
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        //get region of interest for the current coin
        cv::Rect roi(center.x - radius, center.y - radius, radius * 2, radius * 2);
        cv::Mat roi_bg = bg_mask(roi);
        //get a mask for the coin
        coin_mask = cv::Mat::zeros(roi_bg.size(), CV_8UC1);
        cv::circle(coin_mask, cv::Point(radius, radius), radius - 5, 1, -1);
        //find their intersection
        cv::bitwise_and(roi_bg, coin_mask, intersection);
        //count overlap and total pixels
        int pixels_in_coin = cv::countNonZero(coin_mask);
        int bg_pixels = cv::countNonZero(intersection);
        //determine if the circle is extraneous
        //if it is, find it's average hue (for penny classification)
        if(bg_pixels <= pixels_in_coin * bg_ratio)
        {
            new_circles.push_back(circles[i]);
            int location = center.y * src.rows + center.x;
            cv::Mat roi_src = src(roi);
            cv::Mat hsv_src;
            cv::cvtColor(roi_src, hsv_src, cv::COLOR_BGR2HSV);
            int hue_total = 0;
            for(int r = 0; r < roi_src.rows; r++)
                for(int c = 0; c < roi_src.cols; c++)
                    if(coin_mask.at<uchar>(r, c) == 1)
                        hue_total += hsv_src.at<cv::Vec3b>(r, c)[0];
                        
            double avg_hue = hue_total / double(pixels_in_coin);
            max_hue = std::max(max_hue, avg_hue);
            min_hue = std::min(min_hue, avg_hue);
            hue_map[location] = avg_hue;
        }
            
    }
    circles = new_circles;
}
void DrawCircles(cv::Mat &src, std::vector<cv::Vec3f> &circles)
{
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(src, center, radius, cv::Scalar(255, 0, 0), 3);
    }
}

void IdentifyAndDrawCircles(cv::Mat &src, std::vector<cv::Vec3f> &circles, std::unordered_map<int, double> &hues, std::vector<int> &counts)
{
    int &numpennies = counts[0];
    int &numnickels = counts[1];
    int &numdimes = counts[2];
    int &numquarters = counts[3];
    int &numhalf = counts[4];
    double penny_radius_total = 0;
    std::unordered_set<int> pennies;
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int location = center.y * src.rows + center.x;
        double threshhold = min_hue + (max_hue - min_hue) / (0.85 * max_hue / min_hue);
        if(hues[location] < threshhold)
        {
            //printf("hue is ")
            numpennies++;
            penny_radius_total += circles[i][2];
            cv::circle(src, center, cvRound(circles[i][2]), cv::Scalar(0, 255, 0), 10);
            pennies.insert(location);
        }
    }
    double avg_penny_radius = penny_radius_total / numpennies;
    for(int i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        double radius = circles[i][2];
        int location = center.y * src.rows + center.x;
        if(pennies.find(location) != pennies.end()) continue; //don't care about pennies
        double rad_ratio = radius / avg_penny_radius;
        if(rad_ratio <= 1) //meaning it's a dime
        {
            numdimes++;
            cv::circle(src, center, cvRound(radius), cv::Scalar(0, 0, 255), 10);
        }
        else if(rad_ratio <= 1.16)
        {
            numnickels++;
            cv::circle(src, center, cvRound(radius), cv::Scalar(255, 0, 0), 10);
        }
        else if(rad_ratio <= 1.6)
        {
            numquarters++;
            cv::circle(src, center, cvRound(radius), cv::Scalar(0, 255, 255), 10);
        }
        else
        {
            numhalf++;
            cv::circle(src, center, cvRound(radius), cv::Scalar(255, 0, 255), 10);
        }
    }
    // for(int i = 0; i < circles.size(); i++)
    // {
    //     cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    //     int radius = cvRound(circles[i][2]);
    //     cv::circle(src, center, radius, cv::Scalar(255, 0, 0), 3);
    // }
}
void WriteConsoleAndStream(std::ofstream &outfile, char *output)
{
    std::cout << output;
    outfile << output;
}
void WriteResults(std::vector<int> &coinValues, const char *filename)
{
    const char* coinNames[5] = {"Pennies", "Nickels", "Dimes", "Quarters", "Half Dollars"};
    int values[5] = {1, 5, 10, 25, 50};
    int totalCents = 0;
    std::ofstream outfile;
    outfile.open(filename);
    char buffer[100];
    sprintf(buffer, "-----Results-----\n");
    WriteConsoleAndStream(outfile, buffer);
    for(int i = 0; i < 5; i++)
    {
        sprintf(buffer, "%s: %d\n", coinNames[i], coinValues[i]);
        WriteConsoleAndStream(outfile, buffer);
        totalCents += coinValues[i] * values[i];
    }
    sprintf(buffer, "Amount: $%d.%d\n", totalCents / 100, totalCents % 100);
    WriteConsoleAndStream(outfile, buffer);
}
int main(int argc, char** argv)
{ 
    cv::Mat src, src_gray, image;
    const char* inputFile = (argc >= 2) ? argv[1] : "image.jpg";
    LoadImage(inputFile, src, src_gray); //load image
    printf("pass 1\n");
    SmoothImage(src_gray, image);
    DisplayImage("my img", image, cv::Size(1600, 1200));
    cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
    DisplayImage("my img", image, cv::Size(1600, 1200));
    SmoothImage(src_gray, image); //apply morphology
    //get circles from circle detection
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 1, min_circle_dist, canny_thresh, accumulator_thresh, min_radius, max_radius);
    //get an approximate mask for the background color (via histograms)
    cv::Mat bg_mask, display;
    GetBackgroundMask(src, bg_mask);

    cv::Mat inv_mask = 1 - bg_mask;
    src.copyTo(display);
    std::unordered_map<int, double> coinHues; //used for determining if things are pennies
    CleanCircles(circles, src, bg_mask, coinHues);
    //ClassifyCoins(display, circles, pennies, nonpennies);
    //ordering:
    //pennies, nickels, dimes, quarters, half dollars
    std::vector<int> coinsNumbers(5);
    IdentifyAndDrawCircles(display, circles, coinHues, coinsNumbers);
    WriteResults(coinsNumbers, "results.txt");
    cv::imwrite("imagec.jpg", display);
}