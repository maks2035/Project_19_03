#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

cv::RNG rng(12345);

int main()
{
   setlocale(LC_ALL, "Russian");
   std::string A1 = "D:/virandfpc/channels4_profile.jpg";
   std::string A2 = "D:/virandfpc/16.jpg";
   std::string A3 = "D:/virandfpc/17943960.png";
   cv::Mat image = cv::imread(A3);

   if (image.empty()) {
      std::cout << "Ошибка загрузки изображения" << std::endl;
      return -1;
   }

   cv::Mat gauss_im;
   cv::GaussianBlur(image, gauss_im, cv::Size(5, 5), 0);

   cv::Mat gauss;
   cv::GaussianBlur(image, gauss, cv::Size(5, 5), 0);

   cv::Mat gray_image;
   cv::cvtColor(gauss, gray_image, cv::COLOR_BGR2GRAY);

   cv::Mat edges_gray;
   cv::Canny(gauss, edges_gray, 30, 90);

   cv::Mat edges;
   cv::Canny(gauss_im, edges, 30, 90);

   std::vector<std::vector<cv::Point> > contours_gray;
   std::vector<cv::Vec4i> hierarchy_gray;
   findContours(edges_gray, contours_gray, hierarchy_gray, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

   std::vector<std::vector<cv::Point> > contours;
   std::vector<cv::Vec4i> hierarchy;
   findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

   cv::Mat result_gray_contour = cv::Mat::zeros(edges.size(), CV_8UC3);
   for (int i = 0; i < contours_gray.size(); i++) {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      cv::drawContours(result_gray_contour, contours_gray, i, color, 2, cv::LINE_8, hierarchy_gray, 0);
   }

   cv::Mat result_contour = cv::Mat::zeros(edges.size(), CV_8UC3);
   for (int i = 0; i < contours.size(); i++) {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      cv::drawContours(result_contour, contours, i, color, 2, cv::LINE_8, hierarchy, 0);
   }

   cv::Mat orig_result = image.clone();
   for (size_t i = 0; i < contours_gray.size(); i++)
   {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      drawContours(orig_result, contours_gray, i, color, 2, cv::LINE_8, hierarchy, 0);
   }

   cv::Mat result = image.clone();
   for (size_t i = 0; i < contours_gray.size(); i++) {
      double epsilon = 0.001 * arcLength(contours_gray[i], true);
      std::vector<cv::Point> approx;
      approxPolyDP(contours_gray[i], approx, epsilon, true);

      cv::Moments M = moments(approx);
      cv::Point center(M.m10 / M.m00, M.m01 / M.m00);
      cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      if (approx.size() == 3) {
         
         putText(result, "Triangle", center, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      }
      else if (approx.size() == 4) {
         putText(result, "Rectangle", center, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      }
      else {
         putText(result, "Circle", center, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      }
   }

   drawContours(result, contours_gray, -1, cv::Scalar(0, 255, 0), 2);
   cv::imshow("image", image);
   cv::imshow("image_contour", result_contour);
   cv::imshow("image_gray_contour", result_gray_contour);
   cv::imshow("result", result);
   cv::waitKey(0);
   
   return 0;
}

