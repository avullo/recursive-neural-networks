#include "General.h"
#include "Options.h"
#include "StructuredDomain.h"
#include "Node.h"

#include <cassert>
#include <cstring>
#include <iostream>
using namespace std;

/*** Private functions ***/

/* Allocation routines */

void Node::allocFoldingOutputStruct(double*** layers_activations, double** delta_lr) {
  // We assume _lnunits[0.._r-1] vector contains number of
  // output units for each layer in folding part.
  (*layers_activations) = new double*[_r];
  
  for(int k=0; k<_r; k++) {
    (*layers_activations)[k] = new double[_lnunits[k]];
    memset((*layers_activations)[k], 0, _lnunits[k]*sizeof(double));
  }

  // r is representation layer, whose dimension is m
  (*delta_lr) = new double[_lnunits[_r-1]];
  memset((*delta_lr), 0, (_lnunits[_r-1])*sizeof(double));
}

void Node::allocHoutputStruct() {
  // We assume _lnunits[_r.._r+_s-1] vector contains number of
  // output units for each layer in h map.
  _h_layers_activations = new double*[_s];
  
  for(int k=0; k<_s; k++) {
    _h_layers_activations[k] = new double[_lnunits[_r+k]];
    memset(_h_layers_activations[k], 0, _lnunits[_r+k]*sizeof(double));
  }
}

/* Deallocation routines */

void Node::deallocFoldingOutputStruct(double*** layers_activations, double** delta_lr) {
  for(int k=0; k<_r; k++) {
    delete[] (*layers_activations)[k];
    (*layers_activations)[k] = 0;
  }

  delete[] (*layers_activations);
  delete[] (*delta_lr);
  (*layers_activations) = 0; (*delta_lr) = 0;
}

void Node::deallocHoutputStruct() {
  for(int k=0; k<_s; k++) {
    delete[] _h_layers_activations[k];
    _h_layers_activations[k] = 0;
  }

  delete[] _h_layers_activations;
  _h_layers_activations = 0;
}

/* Constructor */
Node::Node(): _norient(num_orientations(Options::instance()->domain())), _outputs(vector<float>(Options::instance()->output_dim(), .0)) {
  // Get fundamental parameters from Options class
  pair<int, int> indexes = Options::instance()->layers_indices();
  _r = indexes.first;
  _s = indexes.second;
  
  _lnunits = Options::instance()->layers_number_units();
  assert(_lnunits.size() == (uint)(_r + _s));

  _ios_tr = (Options::instance()->transduction() == IO_ISOMORPH)?true:false;

  _layers_activations = new double**[_norient];
  _delta_lr = new double*[_norient];
  for(int i=0; i<_norient; ++i)
    allocFoldingOutputStruct(&(_layers_activations[i]), &(_delta_lr[i]));

  if(_ios_tr) 
    allocHoutputStruct();
}

/* Copy Constructor */
Node::Node(const Node& n): _encodedInput(n._encodedInput), _otargets(n._otargets), _outputs(n._outputs) {
  pair<int, int> indexes = Options::instance()->layers_indices();
  _r = indexes.first;
  _s = indexes.second;
  _lnunits = Options::instance()->layers_number_units();
  assert(_lnunits.size() == (unsigned int)(_r + _s));

  if(_otargets.size())
    assert((unsigned int)_lnunits[_r+_s-1] == _otargets.size());

  _ios_tr = (Options::instance()->transduction() == IO_ISOMORPH)?true:false;

  _norient = num_orientations(Options::instance()->domain());
  _layers_activations = new double**[_norient];
  _delta_lr = new double*[_norient];
  for(int i=0; i<_norient; ++i)
    allocFoldingOutputStruct(&(_layers_activations[i]), &(_delta_lr[i]));

  if(_ios_tr) 
    allocHoutputStruct();
}

Node::~Node() {
  for(int i=0; i<_norient; ++i)
    deallocFoldingOutputStruct(&(_layers_activations[i]), &(_delta_lr[i]));
  delete[] _layers_activations; _layers_activations = 0;

  if(_ios_tr)
    deallocHoutputStruct();
}

void Node::resetValues() {
  for(int i=0; i<_norient; ++i) {
    for(int k=0; k<_r; k++)
      memset(_layers_activations[i][k], 0, _lnunits[k]*sizeof(double));

    memset(_delta_lr[i], 0, (_lnunits[_r-1])*sizeof(double));
  }
  
  if(_ios_tr)
    for(int k=0; k<_s; k++)
      memset(_h_layers_activations[k], 0, _lnunits[_r+k]*sizeof(double));
}
