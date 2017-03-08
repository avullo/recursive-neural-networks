#include "catch.hpp"

#include "StructuredDomain.h"
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

TEST_CASE("Basic facs about a domain", "[structured domain]") {
  Domain domains[] = { DOAG, SEQUENCE, LINEARCHAIN, NARYTREE, UG, GRID2D };
  int orientations[] = { 1, 1, 2, 1, 2, 4 };

  for(int i=0; i<6; ++i)
    CHECK(num_orientations(domains[i]) == orientations[i]);
  CHECK_THROWS(num_orientations((Domain)-1));
  CHECK_THROWS(num_orientations((Domain)6));

}
