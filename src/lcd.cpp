#include "lcd.h"
#include <algorithm>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <string>

lcd::lcd(const MapPtr &map, const std::string &netvlad_model) : map_(map) {
  netvlad_torch_ptr_ = std::make_unique<netvlad_torch>(netvlad_model);
}

void lcd::addKeyFrame(FramePtr frame) {
  buffer_mutex_.lock();
  frames_.push(frame);
  buffer_mutex_.unlock();
}

void lcd::loop() {
  std::cout << "start lcd thread!" << std::endl;
  struct timespec ts {};
  ts.tv_sec = 0;           // seconds
  ts.tv_nsec = 500000000L; // 500ms in nanoseconds

  while (!shutdown_) {
    nanosleep(&ts, nullptr); // Sleep for 500 milliseconds
    if (shutdown_) {
      std::cout << "AirVO::lcd::mainLoop: shutdown!" << std::endl;
      break;
    }
    if (frames_.empty()) {
      continue;
    }
    buffer_mutex_.lock();
    auto frm = frames_.front();
    frames_.pop();
    buffer_mutex_.unlock();
    netvlad_torch_ptr_->transform(frm->img_.clone(), frm->getGlobalDesc());

    double highest_score = 0;

    std::map<int, FramePtr> kfs;
    map_->getKeyframe(kfs);
    for (auto &_keyframe : kfs) {
      if (_keyframe.first == frm->GetFrameId()) {
        continue;
      }
      auto score = netvlad_torch::score(frm->getGlobalDesc(),
                                        _keyframe.second->getGlobalDesc());
      if (score > highest_score && score >= 0.6 &&
          std::abs(frm->GetFrameId() - _keyframe.second->GetFrameId()) >= 20) {
        highest_score = score;
        frm->similarity_kf_id_ = _keyframe.first;
      }
    }

    if (highest_score > 0) {
      printf("found highest_score: %f\n", highest_score);
    }

    if (frm->similarity_kf_id_ != -1) {
      auto candidate_kf = kfs.at(frm->similarity_kf_id_);
      auto similar_image = candidate_kf->img_.clone();
      auto frame_image = frm->img_.clone();
      assert(!similar_image.empty() && !frame_image.empty());
      assert(similar_image.rows == frame_image.rows);

      std::vector<cv::DMatch> matches;
      auto frm_feats = frm->getCopyFeatures();
      std::cout << "frm feats " << frm_feats.cols() << " x " << frm_feats.rows()
                << std::endl;
      auto candidate_kf_feats = candidate_kf->getCopyFeatures();
      std::cout << "candidate frm feats " << candidate_kf_feats.cols() << " x "
                << candidate_kf_feats.rows() << std::endl;

      gpu_mutex_.lock();
      auto num_matches = matcher_->MatchingPoints(frm_feats, candidate_kf_feats,
                                                  matches, true, false);
      gpu_mutex_.unlock();

      std::vector<cv::KeyPoint> points0, points1;
      std::vector<int> point_indexes;
      for (size_t i = 0; i < frm_feats.cols(); i++) {
        auto kp = cv::KeyPoint(frm_feats(1, i), frm_feats(2, i), 1);
        points0.emplace_back(kp);
      }

      for (size_t i = 0; i < candidate_kf_feats.cols(); i++) {
        auto kp = cv::KeyPoint(candidate_kf_feats(1, i),
                             candidate_kf_feats(2, i), 1);
        points1.emplace_back(kp);
      }
      cv::Mat out;
      cv::drawMatches(frame_image, points0, similar_image, points1, matches,
                      out);

      // Write the text on the image
      // Define the text properties
      std::string text = std::to_string(frm->GetFrameId()) + " - " +
                         std::to_string(candidate_kf->GetFrameId()) + " - " +
                         std::to_string(highest_score) + " - " +
                         std::to_string(num_matches);

      int fontFace = cv::FONT_HERSHEY_SIMPLEX;
      double fontScale = 1;
      int thickness = 2;
      cv::Point textOrg(
          50, 200); // Bottom-left corner of the text string in the image
      cv::Scalar color(0, 255, 0); // Green color

      cv::putText(out, text, textOrg, fontFace, fontScale, color, thickness);
      std::cout << "matches between frm and candidate: " << num_matches
                << std::endl;
      // Show the concatenated image
      cv::imwrite("/home/vuong/Dev/debug.png", out);
    }
  }
}
