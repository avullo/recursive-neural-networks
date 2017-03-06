#include "Instance.h"
#include <cassert>
using namespace std;

// factory method
Instance* Instance::factory(Enum d, Transduction t, std::istream& is)
  throw(BadInstanceCreation) {
  if(type == SEQUENCE) { return new Sequence(d, t, is); }
}

Instance::Skeleton::Skeleton(Domain domain): _i(-1), _o(-1), _norient(num_orientations(domain)) {
  assert(_norient > 0);
  
  _orientations = new (DPAG*)[_norient];
  _top_orders = new (vector<int>)[_norient];
}

Instance::Skeleton::~Skeleton() {
  for(int i=0; i<_norient; ++i)
    delete _orientations[i];
  delete[] _orientations;
  
  delete[] _top_orders;
}

