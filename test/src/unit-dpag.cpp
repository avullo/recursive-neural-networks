#include "catch.hpp"

#include "DPAG.h"
#include <iostream>

TEST_CASE("DPAG") {
  int num_nodes = 10;
  DPAG d(num_nodes);
  
  CHECK(boost::num_vertices(d) == num_nodes);
  print(d, std::cout);
}
