//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <chrono>

#include "include/config_parser.h"
#include "include/keypoint_detector.h"
#include "include/object_detector.h"
#include "include/preprocess_op.h"
#include "json/json.h"
#include <opencv2/aruco.hpp>

class PaddleLite
{
public:
  PaddleLite() {}

  ~PaddleLite()
  {
    delete keypoint;
    delete det;
  }

  int Init(std::string config_path)
  {
    // Parsing command-line
    PaddleDetection::load_jsonf(config_path, RT_Config);
    if (RT_Config["model_dir_det"].asString().empty())
    {
      std::cout << "Please set [model_det_dir] in " << config_path << std::endl;
      return -1;
    }
    // Load model and create a object detector
    det = new PaddleDetection::ObjectDetector(
        RT_Config["model_dir_det"].asString(),
        RT_Config["cpu_threads"].asInt(),
        RT_Config["batch_size_det"].asInt());

    labels = det->GetLabelList();
    batch_size_det = RT_Config["batch_size_det"].asInt();
    threshold_det = RT_Config["threshold_det"].asFloat();

    keypoint = nullptr;
    if (!RT_Config["model_dir_keypoint"].asString().empty())
    {
      keypoint = new PaddleDetection::KeyPointDetector(
          RT_Config["model_dir_keypoint"].asString(),
          RT_Config["cpu_threads"].asInt(),
          RT_Config["batch_size_keypoint"].asInt(),
          RT_Config["use_dark_decode"].asBool());
      RT_Config["batch_size_det"] = 1;
      printf(
          "batchsize of detection forced to be 1 while keypoint model is not "
          "empty()");
    }
    return 0;
  }

  std::vector<PaddleDetection::ObjectResult> Detection(cv::Mat frame)
  {
    std::vector<cv::Mat> batch_imgs = {frame};
    // Store all detected result
    std::vector<PaddleDetection::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    bool is_rbox = false;
    det->Predict(batch_imgs, threshold_det, 0, 1, &result, &bbox_num, &det_times);

    cv::Mat im = batch_imgs[0];
    std::vector<PaddleDetection::ObjectResult> im_result;
    int detect_num = 0;
    for (int i = 0; i < bbox_num[0]; i++)
    {
      PaddleDetection::ObjectResult item = result[i];
      if (item.confidence < threshold_det || item.class_id == -1)
      {
        continue;
      }
      detect_num += 1;
      im_result.push_back(item);
    }

    return im_result;
  }

  std::vector<PaddleDetection::KeyPointResult>
  Keypoint(cv::Mat im, std::vector<PaddleDetection::ObjectResult> im_result)
  {
    int kpts_imgs = 0;
    // Store keypoint results
    std::vector<PaddleDetection::KeyPointResult> result_kpts;
    std::vector<cv::Mat> imgs_kpts;
    std::vector<std::vector<float>> center_bs;
    std::vector<std::vector<float>> scale_bs;
    int imsize = im_result.size();
    for (int i = 0; i < imsize; i++)
    {
      auto item = im_result[i];
      cv::Mat crop_img;
      std::vector<double> keypoint_times;
      std::vector<int> rect = {
          item.rect[0], item.rect[1], item.rect[2], item.rect[3]};
      std::vector<float> center;
      std::vector<float> scale;
      if (item.class_id == 0)
      {
        PaddleDetection::CropImg(im, crop_img, rect, center, scale);
        center_bs.emplace_back(center);
        scale_bs.emplace_back(scale);
        imgs_kpts.emplace_back(crop_img);
        kpts_imgs += 1;
      }

      if (imgs_kpts.size() == RT_Config["batch_size_keypoint"].asInt() ||
          ((i == imsize - 1) && !imgs_kpts.empty()))
      {
        keypoint->Predict(imgs_kpts,
                          center_bs,
                          scale_bs,
                          0,
                          1,
                          &result_kpts,
                          &keypoint_times);
        imgs_kpts.clear();
        center_bs.clear();
        scale_bs.clear();
      }
    }

    return result_kpts;
  }

public:
  std::vector<std::string> labels;

private:
  int batch_size_det;
  double threshold_det;
  Json::Value RT_Config;
  PaddleDetection::ObjectDetector *det;
  PaddleDetection::KeyPointDetector *keypoint;
};

class Aruco
{
public:
  Aruco()
  {
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    parameters = cv::aruco::DetectorParameters::create();
  }

  ~Aruco() {}

  void Detect(cv::Mat &frame)
  {
    cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
  }

private:
  std::vector<int> markerIds;
  cv::Ptr<cv::aruco::Dictionary> dictionary;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters;
};

namespace demo {
  using namespace std;
  using namespace cv;

  int thresh = 50, N = 11;
  const char *wndname = "Square Detection Demo";

  // helper function:
  // finds a cosine of angle between vectors
  // from pt0->pt1 and from pt0->pt2
  double angle(Point pt1, Point pt2, Point pt0)
  {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
  }

  // returns sequence of squares detected on the image.
  void findSquares(const Mat &image, vector<vector<Point>> &squares)
  {
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point>> contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
      int ch[] = {c, 0};
      mixChannels(&timg, 1, &gray0, 1, ch, 1);

      // try several threshold levels
      for (int l = 0; l < N; l++)
      {
        // hack: use Canny instead of zero threshold level.
        // Canny helps to catch squares with gradient shading
        if (l == 0)
        {
          // apply Canny. Take the upper threshold from slider
          // and set the lower to 0 (which forces edges merging)
          Canny(gray0, gray, 0, thresh, 5);
          // dilate canny output to remove potential
          // holes between edge segments
          dilate(gray, gray, Mat(), Point(-1, -1));
        }
        else
        {
          // apply threshold if l!=0:
          //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
          gray = gray0 >= (l + 1) * 255 / N;
        }

        // find contours and store them all as a list
        findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        vector<Point> approx;

        // test each contour
        for (size_t i = 0; i < contours.size(); i++)
        {
          // approximate contour with accuracy proportional
          // to the contour perimeter
          approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

          // square contours should have 4 vertices after approximation
          // relatively large area (to filter out noisy contours)
          // and be convex.
          // Note: absolute value of an area is used because
          // area may be positive or negative - in accordance with the
          // contour orientation
          if (approx.size() == 4 &&
              fabs(contourArea(approx)) > 1000 &&
              isContourConvex(approx))
          {
            double maxCosine = 0;

            for (int j = 2; j < 5; j++)
            {
              // find the maximum cosine of the angle between joint edges
              double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
              maxCosine = MAX(maxCosine, cosine);
            }

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if (maxCosine < 0.3)
              squares.push_back(approx);
          }
        }
      }
    }
  }

  void findLines(const Mat &image, vector<Vec2f> &lines)
  {
    Mat edges, gray;
    lines.clear();
    cvtColor(image, gray, COLOR_RGB2GRAY);
    Canny(gray, edges, 50, 200, 3);
    HoughLines(edges, lines, 1, CV_PI/180, 250, 0, 0); // runs the actual detection
  }

  void drawLines(const Mat &image, vector<Vec2f> &lines)
  {
    for (size_t i = 0; i < lines.size(); i++)
    {
      float r = lines[i][0], t = lines[i][1];
      double cos_t = cos(t), sin_t = sin(t);
      double x0 = r * cos_t, y0 = r * sin_t;
      double alpha = 1000;

      Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
      Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
      line(image, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
  }

};

int main(int argc, char **argv)
{
  std::string config_path = argv[1];
  std::string video_file = "";
  cv::Mat frame;
  cv::Mat im;
  double tt = 0;
  double fps = 0;
  cv::VideoCapture cap;
  std::vector<PaddleDetection::ObjectResult> im_result;

  if (argc < 2)
  {
    std::cout << "Usage: ./main det_runtime_config.json video" << std::endl;
    return -1;
  }

  if (argc >= 3)
  {
    cap.open(argv[2]);
  }
  else
  {
    cap.open(0);
  }

  cv::Mat vis_img;

  PaddleLite plite;
  plite.Init(config_path);
  Aruco aruco = Aruco();
  int select = 2;

  while (true)
  {
    cap >> frame;
    if (frame.empty())
      break;

    double t = cv::getTickCount();
    cv::resize(frame, im, cv::Size(640, 480));

    switch (select)
    {
      case 0:
      {
        auto colormap = PaddleDetection::GenerateColorMap(plite.labels.size());
        im_result = plite.Detection(im);
        vis_img = PaddleDetection::VisualizeResult(im, im_result, plite.labels, colormap, false);
      };
      break;
      case 1:
      {
        std::vector<std::vector<cv::Point> > squares;
        demo::findSquares(im, squares);
        cv::polylines(im, squares, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        vis_img = im;
      };
      break;
      case 2:
      {
        std::vector<cv::Vec2f> lines;
        demo::findLines(im, lines);
        demo::drawLines(im, lines);
        vis_img = im;
      };
      break;
      case 3:
      {
        aruco.Detect(im);
        vis_img = im;
      };
      break;

      default:
      break;
    }

    tt = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    fps = 1 / tt;

    cv::putText(vis_img, cv::format("FPS = %.2f", fps),
                cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);

    cv::imshow("PaddleLite", vis_img);
    for (auto &item : im_result)
    {
      printf("label: %s, confidence: %.4f\n", plite.labels[item.class_id].c_str(), item.confidence);
      printf("left:%d, right:%d, top:%d, down:%d\n", item.rect[0], item.rect[1], item.rect[2], item.rect[3]);
    }

    int k = cv::waitKey(5);
    if (k == 27)
    {
      cv::destroyAllWindows();
      break;
    }
  }
  return 0;
}
