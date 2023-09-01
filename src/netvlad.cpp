#include "NetVLAD.hpp"

class NetVLAD {
public:
  NetVLAD();
  NetVLAD(NetVLAD &&) = default;
  NetVLAD(const NetVLAD &) = default;
  NetVLAD &operator=(NetVLAD &&) = default;
  NetVLAD &operator=(const NetVLAD &) = default;
  ~NetVLAD();

private:
  
};

NetVLAD::NetVLAD() {
}

NetVLAD::~NetVLAD() {
}
