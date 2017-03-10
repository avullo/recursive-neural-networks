#include "Instance.h"
#include <cassert>
using namespace std;

// factory method
Instance* Instance::factory(istream& is, Domain d, Transduction t, bool read_target)
  throw(BadInstanceCreation) {
  
  if(d == DOAG) { return new DOAG(is, t, read_target); }
  if(d == SEQUENCE) { return new Sequence(is, t, read_target); }
  if(d == LINEARCHAIN) { return new LinearChain(is, t, read_target); }
  if(d == UG) { return new UndirectedGraph(is, t, read_target); }
  if(d == GRID2D) { return new Grid2D(is, t, read_target); }

  throw BadInstanceCreation("Unrecognised domain");
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

void DOAG::read(istream& is, bool target) {
}

void Sequence::read(istream& is, bool target) {
}

void LinearChain::read(istream& is, bool target) {
}

void UndirectedGraph::read(istream& is, bool target) {
}

void Grid2D::read(istream& is, bool target) {
}
