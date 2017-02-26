#include "StructuredDomain.h"
#include <cassert>
using namespace std;

Skeleton::Skeleton(Domain domain): _norient(numOrientations(domain)) {
  assert(_norient > 0);
  
  _orientations = new DPAG[_norient];
  _top_orders = new (vector<int>)[_orient];
}

Skeleton::~Skeleton() {
  delete[] _orientations;
  delete[] _top_orders;
}
