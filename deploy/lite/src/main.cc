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

  if (argc >= 3) {
    cap.open(argv[2]);
  }
  else {
    cap.open(0);
  }

  PaddleLite plite;
  plite.Init(config_path);

  while (true)
  {
    cap >> frame;
    if (frame.empty())
      break;

    double t = cv::getTickCount();
    cv::resize(frame, im, cv::Size(320, 240));
    auto colormap = PaddleDetection::GenerateColorMap(plite.labels.size());
    im_result = plite.Detection(im);
    cv::Mat vis_img = PaddleDetection::VisualizeResult(im, im_result, plite.labels, colormap, false);
    tt = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    fps = 1 / tt;

    cv::putText(vis_img, cv::format("FPS = %.2f", fps),
                cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);

    cv::imshow("PaddleLite", vis_img);
    for (auto &item : im_result) {
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
