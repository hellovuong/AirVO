#include "lcd.h"
#include <algorithm>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

namespace AirVO {
lcd::lcd(const MapPtr &map, const std::string &netvlad_model) : map_(map) {
  netvlad_torch_ptr_ = std::make_unique<netvlad_torch>(netvlad_model);
}

void lcd::addKeyFrame(FramePtr frame) {
  buffer_mutex_.lock();
  frames_.push(frame);
  buffer_mutex_.unlock();
}

void lcd::mainLoop() {
  std::cout << "start lcd thread!" << std::endl;
  struct timespec ts {};
  ts.tv_sec = 0;           // seconds
  ts.tv_nsec = 500000000L; // 500ms in nanoseconds

  while (shutdown_) {
    if (shutdown_) {
      std::cout << "AirVO::lcd::mainLoop: shutdown!" << std::endl;
      break;
    }
    if (frames_.empty()) {
      nanosleep(&ts, nullptr); // Sleep for 500 milliseconds
      continue;
    }
    buffer_mutex_.lock();
    auto frm = frames_.front();
    frames_.pop();
    buffer_mutex_.unlock();
    netvlad_torch_ptr_->transform(frm->img_.clone(), frm->getGlobalDesc());

    if (frm->similarity_kf_id_ != -1) {
      auto similar_image =
          map_->GetFramePtr(frm->similarity_kf_id_)->img_.clone();
      auto frame_image = frm->img_.clone();
      assert(!similar_image.empty() && !frame_image.empty());
      assert(similar_image.rows == frame_image.rows);
      // Concatenate the images
      cv::Mat concatenated_img;
      cv::hconcat(similar_image, frame_image, concatenated_img);

      // Define the text properties
      std::string text =
          std::to_string(frm->GetFrameId()) + " - " +
          std::to_string(
              map_->GetFramePtr(frm->similarity_kf_id_)->GetFrameId());
      int fontFace = cv::FONT_HERSHEY_SIMPLEX;
      double fontScale = 1;
      int thickness = 2;
      cv::Point textOrg(
          50, 200); // Bottom-left corner of the text string in the image
      cv::Scalar color(0, 255, 0); // Green color

      // Write the text on the image
      cv::putText(concatenated_img, text, textOrg, fontFace, fontScale, color,
                  thickness);

      if (concatenated_img.channels() == 1) {
        cv::cvtColor(concatenated_img, concatenated_img, CV_GRAY2BGR);
      }
      // Show the concatenated image
      cv::imwrite("/home/vuong/Dev/debug.png", concatenated_img);
      nanosleep(&ts, nullptr); // Sleep for 500 milliseconds
    }
  }
}
} // namespace AirVO
