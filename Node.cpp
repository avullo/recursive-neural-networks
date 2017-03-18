// #include "require.h"
#include "Node.h"

#include <cassert>
#include <cstring>
#include <iostream>
using namespace std;

//#define WFP int z; cin >> z

/*** Private functions ***/

/* Allocation routines */
void Node::allocStructs(const vector<int>& _lnunits, int _r, int _s, bool _ios_tr, bool _process_dr) {
  allocFoldingOutputStruct(_lnunits, _r, &_f_layers_activations, &_f_delta_lr);
  
  if(_process_dr) {
    allocFoldingOutputStruct(_lnunits, _r, &_b_layers_activations, &_b_delta_lr);
    allocFoldingOutputStruct(_lnunits, _r, &_j_layers_activations, &_j_delta_lr);
    allocFoldingOutputStruct(_lnunits, _r, &_k_layers_activations, &_k_delta_lr);
  }
  
  if(_ios_tr) 
    allocHoutputStruct(_lnunits, _r, _s);
}

void Node::allocFoldingOutputStruct(const vector<int>& _lnunits, int _r, double*** layers_activations, double** delta_lr) {
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

void Node::allocHoutputStruct(const vector<int>& _lnunits, int _r, int _s) {
  // We assume _lnunits[_r.._r+_s-1] vector contains number of
  // output units for each layer in h map.
  _h_layers_activations = new double*[_s];
  
  for(int k=0; k<_s; k++) {
    _h_layers_activations[k] = new double[_lnunits[_r+k]];
    memset(_h_layers_activations[k], 0, _lnunits[_r+k]*sizeof(double));
  }
}

/* Deallocation routines */
void Node::deallocStructs(const vector<int>& _lnunits, int _r, int _s, bool _ios_tr, bool _process_dr) {
  deallocFoldingOutputStruct(_lnunits, _r, &_f_layers_activations, &_f_delta_lr);

  if(_process_dr) {
    deallocFoldingOutputStruct(_lnunits, _r, &_b_layers_activations, &_b_delta_lr);
    deallocFoldingOutputStruct(_lnunits, _r, &_j_layers_activations, &_j_delta_lr);
    deallocFoldingOutputStruct(_lnunits, _r, &_k_layers_activations, &_k_delta_lr);
  }

  if(_ios_tr)
    deallocHoutputStruct(_lnunits, _s);
}

void Node::deallocFoldingOutputStruct(const vector<int>& _lnunits, int _r, double*** layers_activations, double** delta_lr) {
  for(int k=0; k<_r; k++) {
    delete[] (*layers_activations)[k];
    (*layers_activations)[k] = 0;
  }

  delete[] (*layers_activations);
  delete[] (*delta_lr);
  (*layers_activations) = 0; (*delta_lr) = 0;
}

void Node::deallocHoutputStruct(const vector<int>& _lnunits, int _s) {
  for(int k=0; k<_s; k++) {
    delete[] _h_layers_activations[k];
    _h_layers_activations[k] = 0;
  }

  delete[] _h_layers_activations;
  _h_layers_activations = 0;
}

/* Constructor */
Node::Node(const vector<float>& ei): _encodedInput(ei) {
  // // Get fundamental parameters from Options class
  pair<int, int> indexes = Options::instance()->layers_indices();
  int _r = indexes.first, _s = indexes.second;
  vector<int> _lnunits = Options::instance()->layers_number_units();
  assert(_lnunits.size() == (unsigned int)(_r + _s));

  //_ios_tr = (trasd & 1)?true:false;
  bool _ios_tr = (Options::instance()->transduction() == IO_ISOMORPH)?true:false;
  bool _process_dr = true;

  allocFoldingOutputStruct(_lnunits, indexes.first, &_f_layers_activations, &_f_delta_lr);

  if(_process_dr) {
    allocFoldingOutputStruct(_lnunits, indexes.first, &_b_layers_activations, &_b_delta_lr);
    allocFoldingOutputStruct(_lnunits, indexes.first, &_j_layers_activations, &_j_delta_lr);
    allocFoldingOutputStruct(_lnunits, indexes.first, &_k_layers_activations, &_k_delta_lr);
  }

  if(_ios_tr) 
    allocHoutputStruct(_lnunits, indexes.first, indexes.second);
}

/* Copy Constructor */
Node::Node(const Node& n): _encodedInput(n._encodedInput), _otargets(n._otargets) {
  pair<int, int> indexes = Options::instance()->layers_indices();
  int _r = indexes.first, _s = indexes.second;
  vector<int> _lnunits = Options::instance()->layers_number_units();
  assert(_lnunits.size() == (unsigned int)(_r + _s));

  if(_otargets.size())
    assert((unsigned int)_lnunits[_r+_s-1] == _otargets.size());

  bool _ios_tr = (Options::instance()->transduction() == IO_ISOMORPH)?true:false;
  bool _process_dr = true;

  allocFoldingOutputStruct(_lnunits, indexes.first, &_f_layers_activations, &_f_delta_lr);

  if(_process_dr) {
    allocFoldingOutputStruct(_lnunits, indexes.first, &_b_layers_activations, &_b_delta_lr);
    allocFoldingOutputStruct(_lnunits, indexes.first, &_j_layers_activations, &_j_delta_lr);
    allocFoldingOutputStruct(_lnunits, indexes.first, &_k_layers_activations, &_k_delta_lr);
  }

  if(_ios_tr)
    allocHoutputStruct(_lnunits, indexes.first, indexes.second);
}

Node::~Node() {
  pair<int, int> indexes = Options::instance()->layers_indices();
  vector<int> _lnunits = Options::instance()->layers_number_units();
  bool _ios_tr = (Options::instance()->transduction() == IO_ISOMORPH)?true:false;
  bool _process_dr = true;

  deallocFoldingOutputStruct(_lnunits, indexes.first, &_f_layers_activations, &_f_delta_lr);

  if(_process_dr) {
    deallocFoldingOutputStruct(_lnunits, indexes.first, &_b_layers_activations, &_b_delta_lr);
    deallocFoldingOutputStruct(_lnunits, indexes.first, &_j_layers_activations, &_j_delta_lr);
    deallocFoldingOutputStruct(_lnunits, indexes.first, &_k_layers_activations, &_k_delta_lr);
  }

  if(_ios_tr)
    deallocHoutputStruct(_lnunits, indexes.second);
}

void Node::resetValues() {
  pair<int, int> indexes = Options::instance()->layers_indices();
  int _r = indexes.first, _s = indexes.second;
  vector<int> _lnunits = Options::instance()->layers_number_units();
  bool _ios_tr = (Options::instance()->transduction() == IO_ISOMORPH)?true:false;
  bool _process_dr = true;

  for(int k=0; k<_r; k++) {
    memset(_f_layers_activations[k], 0, _lnunits[k]*sizeof(double));
  }
  memset(_f_delta_lr, 0, (_lnunits[_r-1])*sizeof(double));

  if(_process_dr) {
    for(int k=0; k<_r; k++) {
      memset(_b_layers_activations[k], 0, _lnunits[k]*sizeof(double));
      memset(_j_layers_activations[k], 0, _lnunits[k]*sizeof(double));
      memset(_k_layers_activations[k], 0, _lnunits[k]*sizeof(double));
    }
    memset(_b_delta_lr, 0, (_lnunits[_r-1])*sizeof(double));
    memset(_j_delta_lr, 0, (_lnunits[_r-1])*sizeof(double));
    memset(_k_delta_lr, 0, (_lnunits[_r-1])*sizeof(double));
  }

  if(_ios_tr) {
    for(int k=0; k<_s; k++) {
      memset(_h_layers_activations[k], 0, _lnunits[_r+k]*sizeof(double));
    }
  }
}
