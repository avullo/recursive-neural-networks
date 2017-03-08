#include "StructuredDomain.h"
using namespace std;

int num_orientations(Domain domain) throw(logic_error) {
  switch(domain) {
  case DOAG: return 1;
  case SEQUENCE: return 1;
  case LINEARCHAIN: return 2;
  case NARYTREE: return 1;
  case UG: return 2;
  case GRID2D: return 4;
  default: break;
  }

  throw logic_error("Unrecognised domain type");
}
