#ifndef LCD_H_
#define LCD_H_

#include "frame.h"
#include "map.h"
#include "NetVLAD.hpp"
#include <queue>

namespace AirVO {
class lcd {
public:
  lcd(const MapPtr &map, const std::string& netvlad_model);
  lcd(lcd &&) = default;
  lcd(const lcd &) = default;
  lcd &operator=(lcd &&) = default;
  lcd &operator=(const lcd &) = default;
  ~lcd() = default;

  void addKeyFrame(FramePtr frame);

  void mainLoop();

  void shutDown() { shutdown_ = true; }

private:
  std::mutex buffer_mutex_;
  bool shutdown_ = false;
  const MapPtr &map_;
  std::unique_ptr<netvlad_torch> netvlad_torch_ptr_{};
  std::queue<FramePtr> frames_;
};
} // namespace AirVO
#endif
