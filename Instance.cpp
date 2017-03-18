#include "Instance.h"
#include <cassert>
using namespace std;

Instance::Skeleton::Skeleton(Domain domain): _i(-1), _o(-1), _norient(num_orientations(domain)) {
  assert(_norient > 0);
  
  _orientations = new DPAG*[_norient];
  _top_orders = new vector<int>[_norient];
  
  for(uint i=0; i<_norient; ++i)
    _orientations[i] = NULL;

}

Instance::Skeleton::~Skeleton() {
  for(uint i=0; i<_norient; ++i)
    delete _orientations[i];
  delete[] _orientations;  
  delete[] _top_orders;
}


void Instance::Skeleton::orientation(uint index, DPAG* dpag) {
  assert(index>=0 && index<_norient);

  int mi = max_indegree(*dpag);
  int mo = max_outdegree(*dpag);
  if(_i < mi) _i = mi;
  if(_o < mo) _o = mo;

  if(_orientations[index] != NULL) {
    delete _orientations[index];
    
  }
  
  _orientations[index] = dpag;
  
  _top_orders[index] = topological_sort(*(_orientations[index]));
}

const DPAG* Instance::orientation(uint index) {
  assert(index>=0 && index<_skel->_norient);
  
  return _skel->_orientations[index];
}

vector<int> Instance::topological_order(uint index) const {
  assert(index>=0 && index<_skel->_norient);

  return _skel->_top_orders[index];
}
