#pragma once

#include "NetVLAD.hpp"
#include "frame.h"
#include "map.h"
#include "point_matching.h"
#include "read_configs.h"
#include "super_glue.h"
#include <memory>
#include <queue>

class map;
class lcd {
public:
  lcd(const MapPtr &map,
      const std::string &netvlad_model);
  ~lcd() = default;

  void addKeyFrame(FramePtr frame);

  void loop();

  void shutDown() { shutdown_ = true; }

  void setUpMatcher(SuperGlueConfig& config)
  {
    matcher_ = std::make_shared<PointMatching>(config);
  }
private:
  std::mutex buffer_mutex_;
  std::mutex gpu_mutex_;
  bool shutdown_ = false;
  const MapPtr &map_;
  std::unique_ptr<netvlad_torch> netvlad_torch_ptr_{};
  std::queue<FramePtr> frames_;

  PointMatchingPtr matcher_{};
};

