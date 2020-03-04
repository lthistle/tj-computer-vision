#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
int HEIGHT = 500;
int WIDTH = 500;
cv::Scalar EDGE_COLOR(255, 255, 0);
cv::Scalar VERTEX_COLOR(0, 0, 255);
cv::Scalar BG_COLOR(0, 0, 0);
std::string FILENAME = "platonicsolids/dodecahedron.txt";
void rotatePoint(cv::Point3f &pt, double degrees)
{
    double theta = degrees * M_PI / 180.0;
    double s = sin(theta);
    double c = cos(theta);
    double newx = pt.x * c - pt.z * s;
    double newz = pt.x * s + pt.z * c;
    pt.x = newx;
    pt.z = newz;
}
double dist_sq(cv::Point3f &pt1, cv::Point3f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    double dz = pt1.z - pt2.z;
    return dx*dx + dy*dy + dz*dz;
}
int main() {
    //read points from file
    std::vector<cv::Point3f> p3d;
    std::ifstream infile(FILENAME);
    double _x, _y, _z;
    while(infile >> _x >> _y >> _z)
        p3d.push_back(cv::Vec3f(_x, _y, _z));
        
    //calculate min distances between points
    std::vector<double> distances(p3d.size());
    for(int i = 0; i < p3d.size(); i++)
    {
        distances[i] = INFINITY;
        for(int j = 0; j < p3d.size(); j++)
            if(i != j)
                distances[i] = std::min(distances[i], dist_sq(p3d[i], p3d[j]));
    }
    //calculate edges
    std::vector<cv::Vec2i> edges;
    for(int i = 0; i < p3d.size(); i++)
        for(int j = i + 1; j < p3d.size(); j++)
            if(abs(dist_sq(p3d[i], p3d[j]) - distances[i]) < 0.00001)
                edges.push_back(cv::Vec2i(i, j));

    cv::Point3f camera(0, 0, -10);
    cv::Point3f plane(0, 0, camera.z + 2.8);
    double cp_dist = plane.z - camera.z;

    for(int t = 0; t < 360; t++)
    {
        cv::Mat img(HEIGHT, WIDTH, CV_8UC3, BG_COLOR);
        std::vector<cv::Point> p2d;
        for(int i = 0; i < p3d.size(); i++)
        {
            cv::Point3f &pt = p3d[i];
            rotatePoint(pt, 1);
            double dx = pt.x - camera.x;
            double dy = pt.y - camera.y;
            double dz = pt.z - camera.z;
            double mapped_x = (dx / dz) * cp_dist;
            double mapped_y = (dy / dz) * cp_dist;
            int px = mapped_x * WIDTH + WIDTH / 2;
            int py = mapped_y * HEIGHT + HEIGHT / 2;
            p2d.push_back(cv::Point(px, py));
            cv::circle(img, cv::Point(px, py), 3, VERTEX_COLOR, -1);
        }
        for(int i = 0; i < edges.size(); i++)
            cv::line(img, p2d[edges[i][0]], p2d[edges[i][1]], EDGE_COLOR, 2);
        char buffer[100];
        sprintf(buffer, "images/frame%03d.png", t);
        cv::imwrite(buffer, img);
        cv::imshow(FILENAME, img);
        cv::waitKey(10);
    }
}