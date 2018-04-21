#ifndef RECURSIVE_NN_H
#define RECURSIVE_NN_H

#include "require.h"
#include "Options.h"
#include "ActivationFunctions.h"
#include "ErrorMinimizationProcedure.h"
#include "DataSet.h"

#include <ctime>
#include <cfloat>

#include <vector>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

// Generate a random number between 0.0 and 1.0
double rnd01() {
  return ((double) rand() / (double) RAND_MAX);
}

// Generate a random number between -1.0 and +1.0
double nrnd01() {
  return ((rnd01() * 2.0) - 1.0);
}

/*
  Declaration and non-inline definition of the class 
  that represents a RecursiveNN model for structured inputs as DPAGs.
  Future changes will include the implementation of the Bridge
  Pattern to separate network abstraction from multiple
  implementation with different space and time complexity
  (e.g. the present and an implementation based on blitz::array class).
  Another natural change would be to implement different
  error minimization procedures (adding momentum to gradient descent,
  conjugate gradient) via the Strategy Pattern.
*/

template<class HA_Function, class OA_Function, class EMP>
class RecursiveNN {
  
  bool _ss_tr; // implement a super-source transduction
  bool _ios_tr; // implement an io-isomorf structural trasduction
  Problem _problem; // type of learning problem
  
  // number of orientations (i.e. state transition functions)
  // to consider for one particular domain
  int _norient;

  // Useful indices as reported in (Goller §3.2)
  // _r: number of layers of the MLP state transition function(s)
  // _s: number of layers of the MLP output function (super-source or io-isomorph)
  int _n, _v, _m, _q, _r, _s;
  
  /*
    Number of units for each layer in an MLP:

    - 0 .. _r-1: state transition function number of units per layer
    - _r .. _s-1: output function (super-source | io-isomorph) number of units per layer
    
    NOTE: thresholds not included, must be taken into account
  */
  std::vector<int> _lnunits;
  
  /*
    Represent connection weights between successive layers
    for each neural network (MLP) implementing a state transition
    function along an orientation of the data structure
   */
  double**** _layers_w;
  double**** _prev_layers_w; // weight values in previous learning step
  // error signals (delta) for each unit/layer/orientation
  // (exclude representation layer, stored in node) 
  double***  _delta_layers;
  // the contribution to the gradient of each connection weight
  // between successive layers in each MLP
  double**** _layers_gradient_w; 

  typedef double*** Node::*PTNLA;
  typedef double**  Node::*PTNDV;
  PTNLA ptn_la;
  PTNDV ptn_dv;
  
  /*
    Super-source transduction 
    A global output is associated to a given instance

    The output function is again a MLP
   */
  double*** _g_layers_w;
  double*** _prev_g_layers_w;
  double**  _g_layers_activations;
  double**  _delta_g_layers;
  double*** _g_layers_gradient_w;
  
  /*
    IO-Isomorph transduction: an output is associate to each node of a given instance
    Represents connection weights in the network implementing the node output function
   */
  double*** _h_layers_w;
  double*** _prev_h_layers_w;
  double**  _delta_h_layers; // error signals in h output map layers
  double*** _h_layers_gradient_w;
 
  // Template parameters indicate the type of hidden and output units
  // activation function.
  HA_Function haf;
  OA_Function oaf;

  /* 
     Templatized Strategy Pattern. 
     The Net mantains an object that implements one 
     of the possibile error minimization procedures.
  */
  friend EMP; // EMP object must be able to easily update net weights
  EMP _wu_method;

  /* Private functions */

  // Specialized allocation&deallocation functions
  void allocFoldingParts(double****, double****, double****, double***);
  void allocSSPart();
  void allocIOSPart();

  void deallocFoldingParts(double****, double****, double****, double***);
  void deallocSSPart();
  void deallocIOSPart();

  // Reset gradient components every time weights are updated.
  void resetFoldingGradient(double****);
  void resetSSGradient();
  void resetIOSGradient();

  // Reset output values in g MLP layers
  void resetSSValues();
  
  // Propagation routines for
  // each specific part of the Net.
  void propagateInputOnFoldingPart(Instance*, int);
  void gPropagateInput(Instance*);
  void hPropagateInput(Node*);

  // Error Back-Propagation Through Structures 
  // routines for each specific part of the Net.
  void backPropOnFoldingPart(Instance*, int);
  void gBackPropagateError(Instance*);
  void hBackPropagateError(Node*);

  double computeSSError(Instance*);
  double computeIOSError(Instance*);
  
 public:
  /*
    Constructor used to train a RNN.
    Network weights are initialized to random values.
  */
  RecursiveNN();

  /* Constructor: read network parameters from file */
  RecursiveNN(const char*);

  /* Destructor */
  ~RecursiveNN();

  void propagateStructuredInput(Instance*);
  void backPropagateError(Instance*);

  // Implements weight update rule
  void adjustWeights(float = 0.0, float = 0.0, float = 0.0);
  // To restore previous parameters settings, so
  // we can adjust learning rate, in case we make a bad move
  void restorePrevWeights();

  // To reset gradient components, made public so training procedure
  // can use it to begin another training phase using the same network.
  void resetGradientComponents();

  /* 
   * this is better moved at the application level 
   */
  // Evaluate error and performance of the net over a data set
  // float evaluatePerformanceOnDataSet(DataSet*, bool = false);

  // Save learned Network parameters to file
  void saveParameters(const char*);

  // Compute error for a structure
  double computeError(Instance*);
  double computeErrorOnDataset(DataSet*);

  // Compute the (squared norm) of the weights, necessary in order
  // to compute the error when regularization is used (weight decay).
  double computeWeightsNorm();
};



/*********************************************************
  Non-inline template class member functions definitions
*********************************************************/


/* Private: F folding part allocation routine */

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::allocFoldingParts(double**** layers_w, double**** prev_layers_w, double**** layers_gradient_w, double*** delta_layers) {
  // Assume constructor has initialized required dimension quantities
  *layers_w = new double**[_r];
  *prev_layers_w = new double**[_r];
  *layers_gradient_w = new double**[_r];
  if(_r > 1) {
    *delta_layers = new double*[_r-1];
    (*delta_layers)[0] = new double[_lnunits[0]];
    memset((*delta_layers)[0], 0, (_lnunits[0]) * sizeof(double));
  }

  // Allocate weights and gradient components matrixes
  // for f folding part. Connections from input layer
  // have special dimensions.
  (*layers_w)[0] = new double*[(_n+_v*_m) + 1];
  (*prev_layers_w)[0] = new double*[(_n+_v*_m) + 1];
  (*layers_gradient_w)[0] = new double*[(_n+_v*_m) + 1];

  for(int i=0; i<(_n+_v*_m) + 1; i++) {
    (*layers_w)[0][i] = new double[_lnunits[0]];
    (*prev_layers_w)[0][i] = new double[_lnunits[0]];
    (*layers_gradient_w)[0][i] = new double[_lnunits[0]];

    // Reset gradient for corresponding weights
    memset((*layers_gradient_w)[0][i], 0, _lnunits[0] * sizeof(double));
	
    // Assign weights a random number between -1.0 and +1.0
    for(int j=0; j<_lnunits[0]; j++) {
      (*layers_w)[0][i][j] = nrnd01() / static_cast<double>(_lnunits[0]);
      (*prev_layers_w)[0][i][j] = (*layers_w)[0][i][j];
    }
  }
  
  // Allocate f weights&gradient matrixes for f folding part.
  for(int k=1; k<_r; k++) {
    // Allocate space for weight&delta matrix between layer i-1 and i.
    // Automatically include space for threshold unit in layer i-1
    (*layers_w)[k] = new double*[_lnunits[k-1] + 1];
    (*prev_layers_w)[k] = new double*[_lnunits[k-1] + 1];
    (*layers_gradient_w)[k] = new double*[_lnunits[k-1] + 1];

    for(int i=0; i<_lnunits[k-1]+1; i++) {
      (*layers_w)[k][i] = new double[_lnunits[k]];
      (*prev_layers_w)[k][i] = new double[_lnunits[k]];
      (*layers_gradient_w)[k][i] = new double[_lnunits[k]];

      // Reset f gradient components for corresponding weights
      memset((*layers_gradient_w)[k][i], 0, (_lnunits[k])*sizeof(double));

      // Assign f weights a random number between -1.0 and +1.0
      for(int j=0; j<_lnunits[k]; j++) {
	(*layers_w)[k][i][j] = nrnd01() / static_cast<double>(_lnunits[k]);
	(*prev_layers_w)[k][i][j] = (*layers_w)[k][i][j];
      }
    }
   
    if(k < _r-1) {
      (*delta_layers)[k] = new double[_lnunits[k]];
      memset((*delta_layers)[k], 0, (_lnunits[k]) * sizeof(double));
    }
  }  
}

/* Private: SS tranforming part allocation routine */
template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::allocSSPart() {
  // Assume constructor has initialized required dimension quantities
  _g_layers_w = new double**[_s];
  _prev_g_layers_w = new double**[_s];
  _g_layers_gradient_w = new double**[_s];
  _g_layers_activations = new double*[_s];
  _delta_g_layers = new double*[_s];

  // Allocate weights and gradient components matrixes
  // for g transforming part. Connections from input layers
  // have special dimensions.
  _g_layers_w[0] = new double*[_norient*_m + 1];
  _prev_g_layers_w[0] = new double*[_norient*_m + 1];
  _g_layers_gradient_w[0] = new double*[_norient*_m + 1];
  
  // Allocate and reset g layers output activation units and delta values.
  _g_layers_activations[0] = new double[_lnunits[_r]];
  memset(_g_layers_activations[0], 0, (_lnunits[_r])*sizeof(double));
  _delta_g_layers[0] = new double[_lnunits[_r]];
  memset(_delta_g_layers[0], 0, (_lnunits[_r])*sizeof(double));


  for(int i=0; i<_norient*_m + 1; i++) {
    _g_layers_w[0][i] = new double[_lnunits[_r]];
    _prev_g_layers_w[0][i] = new double[_lnunits[_r]];
    _g_layers_gradient_w[0][i] = new double[_lnunits[_r]];

    // Reset gradient for corresponding weights
    memset(_g_layers_gradient_w[0][i], 0, _lnunits[_r] * sizeof(double));
	
    // Assign weights a random number between -1.0 and +1.0
    for(int j=0; j<_lnunits[_r]; j++) {
      _g_layers_w[0][i][j] = nrnd01() / static_cast<double>(_lnunits[_r]);
      _prev_g_layers_w[0][i][j] = _g_layers_w[0][i][j];
    }
  }
  
  // Allocate weights&gradient matrixes for g transforming part
  for(int k=1; k<_s; k++) {
    // Allocate space for weight&gradient matrixes between layer i-1 and i.
    // Automatically include space for threshold unit in layer i-1
    _g_layers_w[k] = new double*[_lnunits[_r+k-1] + 1];
    _prev_g_layers_w[k] = new double*[_lnunits[_r+k-1] + 1];
    _g_layers_gradient_w[k] = new double*[_lnunits[_r+k-1] + 1];

    // Allocate and reset g layers output activation units and delta layers values.
    _g_layers_activations[k] = new double[_lnunits[_r+k]];
    memset(_g_layers_activations[k], 0, (_lnunits[_r+k])*sizeof(double));

    _delta_g_layers[k] = new double[_lnunits[_r+k]];
    memset(_delta_g_layers[k], 0, (_lnunits[_r+k])*sizeof(double));

    for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
      _g_layers_w[k][i] = new double[_lnunits[_r+k]];
      _prev_g_layers_w[k][i] = new double[_lnunits[_r+k]];
      _g_layers_gradient_w[k][i] = new double[_lnunits[_r+k]];

      // Reset g gradient components for corresponding weights
      memset(_g_layers_gradient_w[k][i], 0, (_lnunits[_r+k])*sizeof(double));

      // Assign g weights a random number between -1.0 and +1.0
      for(int j=0; j<_lnunits[_r+k]; j++) {
	_g_layers_w[k][i][j] = nrnd01() / static_cast<double>(_lnunits[_r+k]);
	_prev_g_layers_w[k][i][j] = _g_layers_w[k][i][j];
      }
    } 
  }
}

/* Private: IOS tranforming part allocation routine */
template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::allocIOSPart() {
  // Assume constructor has initialized required dimension quantities
  _h_layers_w = new double**[_s];
  _prev_h_layers_w = new double**[_s];
  _h_layers_gradient_w = new double**[_s];
  _delta_h_layers = new double*[_s];

  // Allocate weights and gradient components matrixes
  // for h transforming part. Connections from input layers
  // have special dimensions.
  _h_layers_w[0] = new double*[_norient*_m + _n + 1];
  _prev_h_layers_w[0] = new double*[_norient*_m + _n + 1];
  _h_layers_gradient_w[0] = new double*[_norient*_m + _n + 1];
  
  // Allocate and reset h first layer delta values.
  _delta_h_layers[0] = new double[_lnunits[_r]];
  memset(_delta_h_layers[0], 0, (_lnunits[_r])*sizeof(double));

  for(int i=0; i<_norient*_m + _n + 1; i++) {
    _h_layers_w[0][i] = new double[_lnunits[_r]];
    _prev_h_layers_w[0][i] = new double[_lnunits[_r]];
    _h_layers_gradient_w[0][i] = new double[_lnunits[_r]];

    // Reset gradient for corresponding weights
    memset(_h_layers_gradient_w[0][i], 0, _lnunits[_r] * sizeof(double));
	
    // Assign weights a random number between -1.0 and +1.0
    for(int j=0; j<_lnunits[_r]; j++) {
      _h_layers_w[0][i][j] = nrnd01() / static_cast<double>(_lnunits[_r]);
      _prev_h_layers_w[0][i][j] = _h_layers_w[0][i][j];
    }
  }
  
  // Allocate weights&gradient matrixes for h map
  for(int k=1; k<_s; k++) {
    // Allocate space for weight&gradient matrixes between layer i-1 and i.
    // Automatically include space for threshold unit in layer i-1
    _h_layers_w[k] = new double*[_lnunits[_r+k-1] + 1];
    _prev_h_layers_w[k] = new double*[_lnunits[_r+k-1] + 1];
    _h_layers_gradient_w[k] = new double*[_lnunits[_r+k-1] + 1];

    // Allocate and reset h layers delta layers values.
    _delta_h_layers[k] = new double[_lnunits[_r+k]];
    memset(_delta_h_layers[k], 0, (_lnunits[_r+k])*sizeof(double));

    for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
      _h_layers_w[k][i] = new double[_lnunits[_r+k]];
      _prev_h_layers_w[k][i] = new double[_lnunits[_r+k]];
      _h_layers_gradient_w[k][i] = new double[_lnunits[_r+k]];

      // Reset h gradient components for corresponding weights
      memset(_h_layers_gradient_w[k][i], 0, (_lnunits[_r+k])*sizeof(double));

      // Assign h weights a random number between -1.0 and +1.0
      for(int j=0; j<_lnunits[_r+k]; j++) {
	_h_layers_w[k][i][j] = nrnd01() / static_cast<double>(_lnunits[_r+k]);
	_prev_h_layers_w[k][i][j] = _h_layers_w[k][i][j];
      }
    }
  }
}


/* Private: Folding parts deallocation routine */
template<class HA_Function, class OA_Function, class EMP> 
void RecursiveNN<HA_Function, OA_Function, EMP>::deallocFoldingParts(double**** layers_w, double**** prev_layers_w, double**** layers_gradient_w, double*** delta_layers) {
  if((*layers_w)[0]) {
    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      if((*layers_w)[0][i]) {
	delete[] (*layers_w)[0][i];
	delete[] (*prev_layers_w)[0][i];
      }
      (*layers_w)[0][i] = 0; (*prev_layers_w)[0][i] = 0;
    }
    delete[] (*layers_w)[0]; delete[] (*prev_layers_w)[0];
    (*layers_w)[0] = 0; (*prev_layers_w)[0] = 0;
  }
  if((*layers_gradient_w)[0]) {
    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      if((*layers_gradient_w)[0][i])
	delete[] (*layers_gradient_w)[0][i];
      (*layers_gradient_w)[0][i] = 0;
    }
    delete[] (*layers_gradient_w)[0];
    (*layers_gradient_w)[0] = 0;
  }

  if(_r > 1 && (*delta_layers)[0]) {
    delete[] (*delta_layers)[0];
    (*delta_layers)[0] = 0;
  }

  for(int k=1; k<_r; k++) {
    if((*layers_w)[k]) {
      for(int i=0; i<_lnunits[k-1]+1; i++) {
	if((*layers_w)[k][i]) {
	  delete[] (*layers_w)[k][i];
	  delete[] (*prev_layers_w)[k][i];
	}
	(*layers_w)[k][i] = 0; (*prev_layers_w)[k][i] = 0;
      }
      delete[] (*layers_w)[k]; delete[] (*prev_layers_w)[k];
      (*layers_w)[k] = 0; (*prev_layers_w)[k] = 0;
    }

    if((*layers_gradient_w)[k]) {
      for(int i=0; i<_lnunits[k-1]+1; i++) {
	if((*layers_gradient_w)[k][i])
	  delete[] (*layers_gradient_w)[k][i];
	(*layers_gradient_w)[k][i] = 0;
      }
      delete[] (*layers_gradient_w)[k];
      (*layers_gradient_w)[k] = 0;
    }
   
    if(k < _r-1 && (*delta_layers)[k]) {
      delete[] (*delta_layers)[k];
      (*delta_layers)[k] = 0;
    }
  }
  
  delete[] *layers_w; delete[] *prev_layers_w;
  delete[] *layers_gradient_w;
  if(_r > 1 && *delta_layers) {
    delete[] *delta_layers;
    *delta_layers = 0;
  }
  *layers_w = 0; *prev_layers_w = 0; *layers_gradient_w = 0;
}

/* Private: SS folding part deallocation routine */
template<class HA_Function, class OA_Function, class EMP> 
void RecursiveNN<HA_Function, OA_Function, EMP>::deallocSSPart() {
  if(_g_layers_w[0]) {
    for(int i=0; i<_norient*_m + 1; i++) {
      if(_g_layers_w[0][i]) {
	delete[] _g_layers_w[0][i];
	delete[] _prev_g_layers_w[0][i];
      }
      _g_layers_w[0][i] = 0; _prev_g_layers_w[0][i] = 0;
    }
    delete[] _g_layers_w[0]; delete[] _prev_g_layers_w[0];
    _g_layers_w[0] = 0; _prev_g_layers_w[0] = 0;
  }
  if(_g_layers_gradient_w[0]) {
    for(int i=0; i<_norient*_m + 1; i++) {
      if(_g_layers_gradient_w[0][i])
	delete[] _g_layers_gradient_w[0][i];
      _g_layers_gradient_w[0][i] = 0;
    }
    delete[] _g_layers_gradient_w[0];
    _g_layers_gradient_w[0] = 0;
  }

  if(_g_layers_activations[0]) {
    delete[] _g_layers_activations[0];
    _g_layers_activations[0] = 0;
  }

  if(_delta_g_layers[0]) {
    delete[] _delta_g_layers[0];
    _delta_g_layers[0] = 0;
  }

  for(int k=1; k<_s; k++) {
    if(_g_layers_w[k]) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	if(_g_layers_w[k][i]) {
	  delete[] _g_layers_w[k][i];
	  delete[] _prev_g_layers_w[k][i];
	}
	_g_layers_w[k][i] = 0; _prev_g_layers_w[k][i] = 0;
      }
      delete[] _g_layers_w[k]; delete[] _prev_g_layers_w[k]; 
      _g_layers_w[k] = 0; _prev_g_layers_w[k] = 0; 
    }

    if(_g_layers_gradient_w[k]) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	if(_g_layers_gradient_w[k][i])
	  delete[] _g_layers_gradient_w[k][i];
	_g_layers_gradient_w[k][i] = 0;
      }
      delete[] _g_layers_gradient_w[k];
      _g_layers_gradient_w[k] = 0;
    }
    
    if(_g_layers_activations[k]) {
      delete[] _g_layers_activations[k];
      _g_layers_activations[k] = 0;
    }

    if(_delta_g_layers[k]) {
      delete[] _delta_g_layers[k];
      _delta_g_layers[k] = 0;
    }
  }
  
  delete[] _g_layers_w; delete[] _prev_g_layers_w;
  delete[] _g_layers_gradient_w;
  delete[] _g_layers_activations;
  delete[] _delta_g_layers;
  _g_layers_w = 0; _prev_g_layers_w = 0; _g_layers_gradient_w = 0;
  _g_layers_activations = 0; _delta_g_layers = 0;

}


/* Private: IOS output map deallocation routine */
template<class HA_Function, class OA_Function, class EMP> 
void RecursiveNN<HA_Function, OA_Function, EMP>::deallocIOSPart() {
  if(_h_layers_w[0]) {
    for(int i=0; i<_norient*_m + _n + 1; i++) {
      if(_h_layers_w[0][i]) {
	delete[] _h_layers_w[0][i];
	delete[] _prev_h_layers_w[0][i];
      }
      _h_layers_w[0][i] = 0; _prev_h_layers_w[0][i] = 0;
    }
    delete[] _h_layers_w[0]; delete[] _prev_h_layers_w[0];
    _h_layers_w[0] = 0; _prev_h_layers_w[0] = 0;
  }
  if(_h_layers_gradient_w[0]) {
    for(int i=0; i<_norient*_m + _n + 1; i++) {
      if(_h_layers_gradient_w[0][i])
	delete[] _h_layers_gradient_w[0][i];
      _h_layers_gradient_w[0][i] = 0;
    }
    delete[] _h_layers_gradient_w[0];
    _h_layers_gradient_w[0] = 0;
  }

  if(_delta_h_layers[0]) {
    delete[] _delta_h_layers[0];
    _delta_h_layers[0] = 0;
  }

  for(int k=1; k<_s; k++) {
    if(_h_layers_w[k]) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	if(_h_layers_w[k][i]) {
	  delete[] _h_layers_w[k][i];
	  delete[] _prev_h_layers_w[k][i];
	}
	_h_layers_w[k][i] = 0; _prev_h_layers_w[k][i] = 0;
      }
      delete[] _h_layers_w[k]; delete[] _prev_h_layers_w[k]; 
      _h_layers_w[k] = 0; _prev_h_layers_w[k] = 0; 
    }

    if(_h_layers_gradient_w[k]) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	if(_h_layers_gradient_w[k][i])
	  delete[] _h_layers_gradient_w[k][i];
	_h_layers_gradient_w[k][i] = 0;
      }
      delete[] _h_layers_gradient_w[k];
      _h_layers_gradient_w[k] = 0;
    }

    if(_delta_h_layers[k]) {
      delete[] _delta_h_layers[k];
      _delta_h_layers[k] = 0;
    }
  }
  
  delete[] _h_layers_w; delete[] _prev_h_layers_w;
  delete[] _h_layers_gradient_w;
  delete[] _delta_h_layers;
  _h_layers_w = 0; _prev_h_layers_w = 0; _h_layers_gradient_w = 0;
  _delta_h_layers = 0;
}

/*** Gradient components resetting methods ***/
template<class HA_Function, class OA_Function, class EMP>
  void RecursiveNN<HA_Function, OA_Function, EMP>::resetGradientComponents() {

  for(int i=0; i<_norient; ++i)
    resetFoldingGradient(&(_layers_gradient_w[i]));

  if(_ss_tr)
    resetSSGradient();

  if(_ios_tr)
    resetIOSGradient();
}

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::resetFoldingGradient(double**** layers_gradient_w) {
  for(int i=0; i<(_n+_v*_m) + 1; i++)
    memset((*layers_gradient_w)[0][i], 0, _lnunits[0] * sizeof(double));

  for(int k=1; k<_r; k++)
    for(int i=0; i<_lnunits[k-1]+1; i++)
      memset((*layers_gradient_w)[k][i], 0, (_lnunits[k])*sizeof(double));

}

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::resetSSGradient() {
  for(int i=0; i<_norient*_m + 1; i++)
    memset(_g_layers_gradient_w[0][i], 0, _lnunits[_r] * sizeof(double));

  for(int k=1; k<_s; k++)
    for(int i=0; i<_lnunits[_r+k-1]+1; i++)
      memset(_g_layers_gradient_w[k][i], 0, (_lnunits[_r+k])*sizeof(double));

}

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::resetIOSGradient() {
  for(int i=0; i<_norient*_m + _n + 1; i++)
    memset(_h_layers_gradient_w[0][i], 0, _lnunits[_r] * sizeof(double));

  for(int k=1; k<_s; k++)
    for(int i=0; i<_lnunits[_r+k-1]+1; i++)
      memset(_h_layers_gradient_w[k][i], 0, (_lnunits[_r+k])*sizeof(double));
}


// Reset output values in g MLP layers
template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::resetSSValues() {
  for(int k=0; k<_s; k++) {
    memset(_g_layers_activations[k], 0, (_lnunits[_r+k])*sizeof(double));
  }
}


/****** Public member functions ******/
/*
  Constructor
    - n: dimension for an input to a node
    - v: valence of the domain (maximum outdegree)
    - r: number of layers for the folding part
    - s: number of layers for the tranforming part
    - n_lunits: array where each slot indicates 
    number of units in each layer
    lnunits[r-1] <=> m, number of units in representation layer
    lnunits[r+s-1] <=> q, number of units in g&|h output layer
    
    Obs: the input layer is constituted by n+v*m units grouped
    into a label part and parts for substructures.
*/
template<class HA_Function, class OA_Function, class EMP> RecursiveNN<HA_Function, OA_Function, EMP>::RecursiveNN():
  _norient(num_orientations(Options::instance()->domain())),
    _n(Options::instance()->input_dim()),
    _v(Options::instance()->domain_outdegree()),
    _lnunits(Options::instance()->layers_number_units()) {
    
  // Get other fundamental parameters from Options class
  std::pair<int, int> indexes = Options::instance()->layers_indices();
  _r = indexes.first; _s = indexes.second;
  require(_lnunits.size() == (uint)(_r+_s), "Dim.Error");
  
  // Number of output units of the 
  // transforming part (g MLP ouput function)
  _q = _lnunits.back();

  // _r is the number of layers for f and b folding parts,
  // number of output units (_m) is the same for both.
  _m = _lnunits[_r-1];
  
  // get types of trasductions to implement
  Transduction tr = Options::instance()->transduction();
  switch(tr) {
  case IO_ISOMORPH:
    _ios_tr = true;
    _ss_tr  = false;
    break;
  case SUPER_SOURCE:
    _ios_tr = false;
    _ss_tr  = true;
    break;
  default:
    require(0, "Unknown transduction type");
  }

  _problem = Options::instance()->problem();
  
  // Initialize random number generator
  srand(time(0));

  _layers_w = new double***[_norient];
  _prev_layers_w = new double***[_norient];
  _layers_gradient_w = new double***[_norient];
  _delta_layers = new double**[_norient];
  
  for(int i=0; i<_norient; ++i)
    allocFoldingParts(&(_layers_w[i]), &(_prev_layers_w[i]), &(_layers_gradient_w[i]), &(_delta_layers[i]));

  if(_ss_tr)
    allocSSPart();
  if(_ios_tr)
    allocIOSPart();

  // Allocate weight update method structures using _process_dr
  // to decide whether or not to instantiate its b internal structures.
  _wu_method.setInternals(this);

  ptn_la = &Node::_layers_activations;
  ptn_dv = &Node::_delta_lr;
  
}

/* Constructor */
template<class HA_Function, class OA_Function, class EMP>
  RecursiveNN<HA_Function, OA_Function, EMP>::RecursiveNN(const char* network_filename) {
  std::ifstream is(network_filename);
  assure(is, network_filename);
  
  // First the number of causal transductions and folding processing to implement
  is >> _norient >> _ios_tr >> _ss_tr;
  
  // Read node input dimension and max outdegree with
  // the network was trained (or initialized)
  is >> _n >> _v;
  
  // Perfom necessary synchronization control with global parameters,
  // to prevent using the net with values different from parameters
  // of the other cooperating classes
  Transduction tr = Options::instance()->transduction();
  require(_ios_tr == (tr == IO_ISOMORPH)?true:false && _ss_tr == (tr == SUPER_SOURCE)?true:false,
	  "Synchronization Error between RecursiveNN and global parameters: check trasduction type");
  require(_norient == num_orientations(Options::instance()->domain()), "Mismatch on number of orientations");
  require(_n == Options::instance()->input_dim(), "Synchronization Error between RecursiveNN and global parameters: check nodes input dimension");
  require(_v == Options::instance()->domain_outdegree(), "Synchronization Error between RecursiveNN and global parameters: check outdegree");

  _problem = Options::instance()->problem();
  
  // Then read values of r and s from file
  is >> _r >> _s;
  // Some other necessary controls
  std::pair<int, int> indexes = Options::instance()->layers_indices();
  require(_r == indexes.first && _s == indexes.second,
	  "Synchronization Error between RecursiveNN and global parameters: check layers indexes");

  // Then the vector of number of units per layer
  int i = 0, lnu;
  while(i<_r+_s) {
    is >> lnu;
    _lnunits.push_back(lnu);
    ++i;
  }
  require(_lnunits == Options::instance()->layers_number_units(),
	  "Synchronization Error between RecursiveNN and global parameters: check layers number of units");

  // Number of output units of the transforming part (g MLP ouput function)
  _q = _lnunits.back();

  // _r is the number of layers for the folding parts,
  // number of output units (_m) is the same for all
  _m = _lnunits[_r-1];

  is.precision(Options::instance()->precision());
  is.setf(std::ios::scientific);

  _layers_w = new double***[_norient];
  _prev_layers_w = new double***[_norient];
  _layers_gradient_w = new double***[_norient];
  _delta_layers = new double**[_norient];

  // allocate space for each folding direction
  for(int i=0; i<_norient; ++i)
    allocFoldingParts(&(_layers_w[i]), &(_prev_layers_w[i]), &(_layers_gradient_w[i]), &(_delta_layers[i]));

  // read weights for each folding direction
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      for(int j=0; j<_lnunits[0]; j++) {
	is >> _layers_w[o][0][i][j]; 
	_prev_layers_w[o][0][i][j] = _layers_w[o][0][i][j]; 
      }
    }
  
    for(int k=1; k<_r; k++) {
      for(int i=0; i<_lnunits[k-1]+1; i++) {
	for(int j=0; j<_lnunits[k]; j++) {
	  is >> _layers_w[o][k][i][j];
	  _prev_layers_w[o][k][i][j] = _layers_w[o][k][i][j];
	}
      }
    }
  }
  
  // Eventually read weights for g output function layers
  if(_ss_tr) {
    allocSSPart();
    
    for(int i=0; i<_norient*_m + 1; i++)
      for(int j=0; j<_lnunits[_r]; j++) {
	is >> _g_layers_w[0][i][j];
	_prev_g_layers_w[0][i][j] = _g_layers_w[0][i][j];
      }
  
    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++) {
	  is >> _g_layers_w[k][i][j];
	  _prev_g_layers_w[k][i][j] = _g_layers_w[k][i][j];
	}
      }  
    }
  }

  // Eventually read weights for h map layers
  if(_ios_tr) {
    allocIOSPart();

    for(int i=0; i<_norient*_m + _n + 1; i++)
      for(int j=0; j<_lnunits[_r]; j++) {
	is >> _h_layers_w[0][i][j];
	_prev_h_layers_w[0][i][j] = _h_layers_w[0][i][j];
      }
  
    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++) {
	  is >> _h_layers_w[k][i][j];
	  _prev_h_layers_w[k][i][j] = _h_layers_w[k][i][j];
	}
      }
    }
  }

  // Allocate weight update method structures
  _wu_method.setInternals(this);

  ptn_la = &Node::_layers_activations;
  ptn_dv = &Node::_delta_lr;

}


/* Destructor */
template<class HA_Function, class OA_Function, class EMP>
RecursiveNN<HA_Function, OA_Function, EMP>::~RecursiveNN() {
  for(int i=0; i<_norient; ++i)
    deallocFoldingParts(&(_layers_w[i]), &(_prev_layers_w[i]), &(_layers_gradient_w[i]), &(_delta_layers[i]));

  if(_ss_tr)
    deallocSSPart();
  
  if(_ios_tr)
    deallocIOSPart();
  
}


/*
  This function is to be used to evaluate a structure in
  a casual or casual & non-casual fashion, compute its encoded 
  representation at root node ouput layers and evaluate it
  with the ouptput (g) function.
*/
template<class HA_Function, class OA_Function, class EMP>
  void RecursiveNN<HA_Function, OA_Function, EMP>::propagateStructuredInput(Instance* instance) {  
  // Reset output activations in nodes layers
  // if _ios_tr is set h output activations are reset
  instance->resetNodeOutputActivations();

  // Reset output activations in output (g) function MLP layers
  if(_ss_tr)
    resetSSValues();

  // Structure propagation by unfolding into casual parts
  for(int i=0; i<_norient; ++i)
    propagateInputOnFoldingPart(instance, i); //toNodes, sdags[0], sdags_top_ords[0], &_f_layers_w, ptn_fla);
  
  // Evaluate current encoded structure (if supersource trasd.)
  if(_ss_tr)
    gPropagateInput(instance); // toNodes, sdags, sdags_top_ords);

  // Compute output label for each node (if io-isomorf trasd.)
  if(_ios_tr)
    for(uint n=0; n<instance->num_nodes(); ++n)
      hPropagateInput(instance->node(n));

}


/*******************
 * Private methods *
 *******************/

template<class HA_Function, class OA_Function, class EMP>
  void RecursiveNN<HA_Function, OA_Function, EMP>::propagateInputOnFoldingPart(Instance* instance, int o) {
  // A structured input is a sequence of nodes and a DAG with that vertices.
  // Nodes are passed by (non-const) reference to allow the net
  // storing output activations on each node for all of the folding layers. 

  DPAG* dpag = instance->orientation(o);
    
  Vertex_d currentNode;
  //VertexId vertex_id = boost::get(boost::vertex_index, *dpag);
  EdgeId edge_id = boost::get(boost::edge_index, *dpag);
  outIter out_i, out_end;

  std::vector<int> top_ord = instance->topological_order(o);
  
  for(std::vector<int>::const_reverse_iterator r_it=top_ord.rbegin(); r_it!=top_ord.rend(); ++r_it) {
    int t = *r_it;

    Node* node = instance->node(t);
    currentNode = boost::vertex(t, *dpag);
    require(_n == node->input_dim(), "Error in Node input dimension\n");

    // Remember: if k==1 (0 according to the indexing scheme) 
    // net input for each unit comes both from current node 
    // immediate successors and from the node input label.

    // We are in layer k == 1.
    // for each unit in layer 1 (0)
    for(int j=0; j<_lnunits[0]; j++) {
      // calculate weighted sum of its inputs
      double unit_input = 0.0;
      // firstly take into account current node input label
      for(int i=0; i<_n; i++) {
	unit_input += 
	  _layers_w[o][0][i][j] * node->_encodedInput[i];
      }

      // add to weighted sum contribution of the previously
      // computed representations of the immediate substructures.
      // Eliminate control on max outdegree.
      // Ignore edges whose id is greater than max outdegree.
      for(boost::tie(out_i, out_end)=out_edges(currentNode, *dpag); 
	  out_i!=out_end && edge_id[*out_i] < (uint)_v; ++out_i) {
	//require(0<=edge_id[*out_i] && edge_id[*out_i] < _v, "Valence assertion failed!");
	Node* successor = instance->node(target(*out_i, *dpag));

	for(uint i=_n + edge_id[*out_i]*_m; i<_n + _m*(edge_id[*out_i] + 1); i++)
	  unit_input +=
	    _layers_w[o][0][i][j] *
	    successor->_layers_activations[o][_r-1][(i-_n)%_m];
	  // or (instance->node(target(*out_i, *dpag))->*ptn_la)[o][_r-1][(i-_n)%_m];
	
      }

      // if there are less children than the valence, missing children encoding 
      // (base step of recursion, 0) does not influence current unit input.
      // So do not add 0 to the sum.
      
      // Add threshold unit contribution (input == 1).
      // We assume threshold unit weight is last component
      // of the weight matrix.
      unit_input += _layers_w[o][0][_n+_v*_m][j];

      // calculate unit output activation
      node->_layers_activations[o][0][j] = evaluate(haf, unit_input);
      // or (node->*ptn_la)[o][0][j] = evaluate(haf, unit_input);
    }

    for(int k=1; k<_r; k++) {
      // for each unit in layer k
      for(int j=0; j<_lnunits[k]; j++) {
	// calculate weighted sum of its input
	double unit_input = 0.0;
	for(int i=0; i<_lnunits[k-1]; i++) {
	  unit_input += _layers_w[o][k][i][j] * node->_layers_activations[o][k-1][i];
	}
	// Add threshold unit contribution (input == 1),
	// last component of the weight matrix.
	unit_input += _layers_w[o][k][_lnunits[k-1]][j];

	// calculate unit output activation
	node->_layers_activations[o][k][j] = evaluate(haf, unit_input);
      }
    }
  }
}

template<class HA_Function, class OA_Function, class EMP>
  void RecursiveNN<HA_Function, OA_Function, EMP>::gPropagateInput(Instance* instance) {
  
  for(int j=0; j<_lnunits[_r]; j++) {
    // calculate weighted sum of its inputs
    double unit_input = 0.0;

    // add to weighted sum contribution of the previously
    // computed representations of the super-source nodes
    // defined for each possible orientation
    for(int o=0; o<_norient; ++o) {
      // the first node in the topological order of each orientation
      // is a super-source node which contributes to the activation
      // of the units of the MLP implementing the super-source transduction
      Node* node = instance->node((instance->topological_orders())[o][0]);
      for(int i=o*_m; i<(o+1)*_m; ++i)
	unit_input +=
	  _g_layers_w[0][i][j] * node->_layers_activations[o][_r-1][i-o*_m];
    }
    
    // Add threshold unit contribution (input == 1).
    // We assume threshold unit weight is last component
    // of the weight matrix.
    unit_input += _g_layers_w[0][_norient*_m][j];

    // calculate units output activation
    if(0 < _s-1) 
      _g_layers_activations[0][j] = evaluate(haf, unit_input);
    else
      _g_layers_activations[0][j] = evaluate(oaf, unit_input);
  }

  for(int k=1; k<_s; k++) {
    // for each unit in layer k
    for(int j=0; j<_lnunits[_r+k]; j++) {
      // calculate weighted sum of its input
      double unit_input = 0.0;
      for(int i=0; i<_lnunits[_r+k-1]; i++) {
	unit_input += 
	  _g_layers_w[k][i][j] * _g_layers_activations[k-1][i];
      }

      // Add threshold unit contribution (input == 1),
      // last component of the weight matrix.
      unit_input += _g_layers_w[k][_lnunits[_r+k-1]][j];

      // calculate unit output activation,
      // take into account being in hidden or output units.
      if(k < _s-1)
	_g_layers_activations[k][j] = evaluate(haf, unit_input);
      else
	_g_layers_activations[k][j] = evaluate(oaf, unit_input);
    }
  }
}


template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::hPropagateInput(Node* n) {
  
  for(int j=0; j<_lnunits[_r]; j++) {
    // calculate weighted sum of its inputs
    double unit_input = 0.0;

    // add to weighted sum contribution of the previously
    // computed representations of the node defined for
    // each possible orientation
    for(int o=0; o<_norient; ++o) {
      for(int i=o*_m; i<(o+1)*_m; ++i)
	unit_input +=
	  _h_layers_w[0][i][j] * n->_layers_activations[o][_r-1][i-o*_m];
    }

    // Output label depend also on current node encoded input
    for(int i=_norient*_m; i<_norient*_m + _n; i++)
      unit_input +=
	_h_layers_w[0][i][j] * n->_encodedInput[i-_norient*_m];

    // Add threshold unit contribution (input == 1).
    // We assume threshold unit weight is last component
    // of the weight matrix.
    unit_input += _h_layers_w[0][_norient*_m+_n][j];

    // calculate units output activation
    if(0 < _s-1) 
      n->_h_layers_activations[0][j] = evaluate(haf, unit_input);
    else
      n->_h_layers_activations[0][j] = evaluate(oaf, unit_input);
  }

  for(int k=1; k<_s; k++) {
    // for each unit in layer k
    for(int j=0; j<_lnunits[_r+k]; j++) {
      // calculate weighted sum of its input
      double unit_input = 0.0;
      for(int i=0; i<_lnunits[_r+k-1]; i++) {
	unit_input += 
	  _h_layers_w[k][i][j] * n->_h_layers_activations[k-1][i];
      }

      // Add threshold unit contribution (input == 1),
      // last component of the weight matrix.
      unit_input += _h_layers_w[k][_lnunits[_r+k-1]][j];

      // calculate unit output activation,
      // take into account being in hidden or output units.
      if(k < _s-1)
	n->_h_layers_activations[k][j] = evaluate(haf, unit_input);
      else
	n->_h_layers_activations[k][j] = evaluate(oaf, unit_input);
    }
  }

}


/*** Public Functions ***/

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::backPropagateError(Instance* instance) {
  // if io-isomorf trasduction, compute for each node error of h map,
  // so as to add it to deltas error in representation layers coming
  // from node parents (with respect to f and b ordering)
  if(_ios_tr)
    for(uint n=0; n<instance->num_nodes(); ++n)
      hBackPropagateError(instance->node(n));

  if(_ss_tr)
    gBackPropagateError(instance);
  
  for(int i=0; i<_norient; ++i)
    backPropOnFoldingPart(instance, i);

}

template<class HA_Function, class OA_Function, class EMP>
  double RecursiveNN<HA_Function, OA_Function, EMP>::computeError(Instance* instance) {

  propagateStructuredInput(instance);
  
  double error = .0;
  if(_ss_tr)
    error += computeSSError(instance);

  if(_ios_tr)
    error += computeIOSError(instance);

  return error;
}

template<class HA_Function, class OA_Function, class EMP>
  double RecursiveNN<HA_Function, OA_Function, EMP>::computeErrorOnDataset(DataSet* dataset) {

  double error = .0;
  for(DataSet::iterator it=dataset->begin(); it!=dataset->end(); ++it)
    error += computeError(*it);

  if(_ss_tr && _problem & REGRESSION)
    error /= dataset->size();

  return error;
}

template<class HA_Function, class OA_Function, class EMP>
  void RecursiveNN<HA_Function, OA_Function, EMP>::backPropOnFoldingPart(Instance* instance, int o) {

  DPAG* dpag = instance->orientation(o);

  Vertex_d currentNode;
  //VertexId vertex_id = boost::get(boost::vertex_index, *dpag);
  EdgeId edge_id = boost::get(boost::edge_index, *dpag);
  outIter out_i, out_end;

  std::vector<int> top_ord = instance->topological_order(o);
  for(std::vector<int>::const_iterator it=top_ord.begin(); it!=top_ord.end(); ++it) {
    int t = *it;

    Node* node = instance->node(t);
    currentNode = boost::vertex(t, *dpag);

    if(_r > 1) {
      /*
       * there are hidden layers:
       * compute their errors using the generalised delta rule
       */
      for(int k=_r-2; k>=0; k--) {
	/*
	 * start at _r-2 as _r-1 is stored at the node level and has already 
	 * been updated with the contribution coming from the node output
	 * network and the deltas coming from the node ancestors
	 */
	memset(_delta_layers[o][k], 0, (_lnunits[k])*sizeof(double));

	for(int i=0; i<_lnunits[k]; i++) {
	  double sum = 0.0;
	  if(k ==_r-2) {
	    /*
	     * this is the penultimate layer, delta values of the preceding 
	     * (representation) layer have been computed and are stored at the node level 
	     */
	    for(int j=0; j<_lnunits[k+1]; j++)
	      sum += _layers_w[o][k+1][i][j] * node->_delta_lr[o][j];
	    // or sum += _layers_w[o][k+1][i][j] * (node->*ptn_dv)[o][j];
	  } else
	    /*
	     * we're in a hidden layer but there are others ahead
	     * so take the delta values directly from them
	     */
	    for(int j=0; j<_lnunits[k+1]; j++)
	      sum += _layers_w[o][k+1][i][j] * _delta_layers[o][k+1][j];
	  
	  _delta_layers[o][k][i] = derivate(haf, node->_layers_activations[o][k][i]) * sum;
	}

	/* 
	 * can now compute the gradient of weights connecting this layer
	 * to the preceding one
	 */
	if(k>0) {
	  /*
	   * this is not the last layer (before the input) of the state transition network
	   * take the activations of the preceding layer from the node
	   */
	  for(int j=0; j<_lnunits[k]; j++) {
	    for(int i=0; i<_lnunits[k-1]; i++)
	      _layers_gradient_w[o][k][i][j] -= 
		_delta_layers[o][k][j] * node->_layers_activations[o][k-1][i];
	    
	    _layers_gradient_w[o][k][_lnunits[k-1]][j] -= _delta_layers[o][k][j];
	  }
	} else {
	  /*
	   * this is the layer connected to the input:
	   * - the current node input
	   * - the representation of the substructures rooted at the children
	   */
	  for(int j=0; j<_lnunits[0]; j++) {
	    // first consider input label
	    for(int i=0; i<_n; i++) {
	      _layers_gradient_w[o][0][i][j] -= 
		_delta_layers[o][k][j] * node->_encodedInput[i];
	    }
	    // then consider ordered children
	    // eliminate control on max outdegree: ignore edges whose id is greater than max outdegree.
	    for(boost::tie(out_i, out_end)=out_edges(currentNode, *dpag); 
		out_i!=out_end && edge_id[*out_i] < (uint)_v; ++out_i) {
	      //require(0<=edge_id[*out_i] && edge_id[*out_i] < _v, "Valence assertion failed!");
	      Node* successor = instance->node(target(*out_i, *dpag));
	      
	      for(uint i=_n + edge_id[*out_i]*_m; i<_n + _m*(edge_id[*out_i] + 1); i++)
		_layers_gradient_w[o][0][i][j] -=
		  _delta_layers[o][k][j] * successor->_layers_activations[o][_r-1][(i-_n)%_m];
	      
	    }
	  }
	}
      }
    } else {
      /*
       * deltas of current layer are already calculated
       * and stored at the node level
       * they've been updated with the errors coming from
       * from the node output network and the contributions
       * coming from the node parent
       */
      for(int j=0; j<_lnunits[0]; j++) {
	// first consider input label
	for(int i=0; i<_n; i++) {
	  _layers_gradient_w[o][0][i][j] -= 
	    node->_delta_lr[o][j] * node->_encodedInput[i];
	}
	// then consider ordered children
	// eliminate control on max outdegree: ignore edges whose id is greater than max outdegree.
	for(boost::tie(out_i, out_end)=out_edges(currentNode, *dpag); 
	    out_i!=out_end && edge_id[*out_i] < (uint)_v; ++out_i) {
	  //require(0<=edge_id[*out_i] && edge_id[*out_i] < _v, "Valence assertion failed!");
	  Node* successor = instance->node(target(*out_i, *dpag));
	  
	  for(uint i=_n + edge_id[*out_i]*_m; i<_n + _m*(edge_id[*out_i] + 1); i++)
	    _layers_gradient_w[o][0][i][j] -=
	      node->_delta_lr[o][j] * successor->_layers_activations[o][_r-1][(i-_n)%_m];

	}
      }
    }

    /*
     * distribute delta error among representation layers
     * of immediate successors of current node t
     */
    for(boost::tie(out_i, out_end)=out_edges(currentNode, *dpag); 
	out_i!=out_end && edge_id[*out_i] < (uint)_v; ++out_i) {
      Node* successor = instance->node(target(*out_i, *dpag));
      
      for(uint i=_n + edge_id[*out_i]*_m; i<_n + _m*(edge_id[*out_i] + 1); i++) {
	double sum = 0.0;
	
	if(_r == 1)
	  /*
	   * one layer in the state transition function
	   * its delta values are stored at the node level and
	   * represent the errors of the following layer to
	   * redistribute to the representation layers of the children 
	   */
	  for(int j=0; j<_lnunits[0]; j++)
	    sum += _layers_w[o][0][i][j] * node->_delta_lr[o][j];
	
	else
	  /*
	   * network has more than one layer hence we take the deltas
	   * at the last one before the input
	   */
	  for(int j=0; j<_lnunits[0]; j++)
	    sum += _layers_w[o][0][i][j] * _delta_layers[o][0][j];

	/* 
	 * delta values for the representation layer of a node t
	 * coming from different immediate predecessors have to be
	 * summed up, before they are propagated deeper into the 
	 * folding part of t and of its successors
	 */
	successor->_delta_lr[o][(i-_n)%_m] +=
	  derivate(haf, successor->_layers_activations[o][_r-1][(i-_n)%_m]) * sum;
      }
    }
  }

}

template<class HA_Function, class OA_Function, class EMP>
  void RecursiveNN<HA_Function, OA_Function, EMP>::gBackPropagateError(Instance* instance) {
  
  int k = _s-1;

  std::vector<float> targets = instance->target();
  std::vector<float> outputs(_g_layers_activations[_s-1],
			     _g_layers_activations[_s-1]+_lnunits[_r+k]);
  require(targets.size() == outputs.size(), "output dim. error");

  /*
   * apply softmax in case of a multi-class (N>2) classification problem
   */
  if(_problem & MULTICLASS) {
    float max = -FLT_MAX;
    for(uint i=0; i<outputs.size(); ++i)
      if(max < outputs[i]) max = outputs[i];
      
    float norm_factor = 0.0;
    for(uint j=0; j<outputs.size(); ++j) {
      outputs[j] = exp(outputs[j] - max);
      norm_factor += outputs[j];
    }
    for(uint j=0; j<outputs.size(); ++j)
      outputs[j] /= norm_factor;
  }

  /*
   * compute errors at the output layer and gradient of
   * the weights from the previous to the output layer
   */
  memset(_delta_g_layers[k], 0, (_lnunits[_r+k])*sizeof(double));

  for(int j=0; j<_lnunits[_r+k]; j++) {
    _delta_g_layers[k][j] =
      (_problem & ~(BINARYCLASS | MULTICLASS)?derivate(oaf, outputs[j]):1.0) *
      (targets[j] - outputs[j]);
        
    if(k>0) {
      /*
       * super-source network has hidden layers:
       * gradient at output units depends on delta and 
       * activation at previous (hidden) layer
       */
      for(int i=0; i<_lnunits[_r+k-1]; i++)
	_g_layers_gradient_w[k][i][j] -= 
	  _delta_g_layers[k][j] * _g_layers_activations[k-1][i];
      
      _g_layers_gradient_w[k][_lnunits[_r+k-1]][j] -= // bias
	_delta_g_layers[k][j];

    } else {
      /*
       * super-source network has just one output layer:
       * activations to compute the gradient come from the activation of the output units
       * of the different state transition networks of the root nodes in each orientation
       */
      for(int o=0; o<_norient; ++o) {
	// the first node in the topological order of each orientation
	// is a super-source node which contributes to the activation
	// of the units of the MLP implementing the super-source transduction
	Node* node = instance->node((instance->topological_orders())[o][0]);
	for(int i=o*_m; i<(o+1)*_m; ++i)
	  _g_layers_gradient_w[0][i][j] -=
	    _delta_g_layers[k][j] * node->_layers_activations[o][_r-1][i-o*_m];
      }

      _g_layers_gradient_w[0][_norient*_m][j] -= _delta_g_layers[k][j];
    }
  }

  /*
   * backpropagate error and compute gradient of the weights in the previous layers
   */
  for(--k; k>=0; k--) {
    memset(_delta_g_layers[k], 0, (_lnunits[_r+k])*sizeof(double));

    for(int i=0; i<_lnunits[_r+k]; i++) {
      double sum = 0.0;
      for(int j=0; j<_lnunits[_r+k+1]; j++) {
	sum += _g_layers_w[k+1][i][j] * _delta_g_layers[k+1][j];
      }
      _delta_g_layers[k][i] = 
	derivate(haf, _g_layers_activations[k][i]) * sum;
    }

    if(k>0) {
      /*
       * this is a hidden layer but there are more:
       * the activations for the delta rule come from
       * the activations of the units from the previous layer
       */
      for(int j=0; j<_lnunits[_r+k]; j++) {
	for(int i=0; i<_lnunits[_r+k-1]; i++)
	  _g_layers_gradient_w[k][i][j] -= 
	    _delta_g_layers[k][j] * _g_layers_activations[k-1][i];

	_g_layers_gradient_w[k][_lnunits[_r+k-1]][j] -=  _delta_g_layers[k][j];
      }
    } else {
      /*
       * this is the last hidden layer before the inputs whose activations come 
       * from the output unit values of the different state transition networks 
       * at the root nodes of each orientation
       */
      for(int j=0; j<_lnunits[_r+k]; j++) {
	for(int o=0; o<_norient; ++o) {
	  Node* node = instance->node((instance->topological_orders())[o][0]);
	  for(int i=o*_m; i<(o+1)*_m; ++i)
	    _g_layers_gradient_w[0][i][j] -=
	      _delta_g_layers[k][j] * node->_layers_activations[o][_r-1][i-o*_m];
	}
	
	_g_layers_gradient_w[0][_norient*_m][j] -= _delta_g_layers[k][j];

      }
    }
  }

  /*
   * calculate the error on input layer and redistribute on the 
   * representation layers of the root nodes of each orientation
   */
  for(int o=0; o<_norient; ++o) {
    Node* n = instance->node((instance->topological_orders())[o][0]);
    
    for(int i=0; i<_m; i++) {
      double sum = 0.0;
      for(int j=0; j<_lnunits[_r]; j++)
	sum += _g_layers_w[0][o*_m + i][j] * _delta_g_layers[0][j];
      
      n->_delta_lr[o][i] += 
	derivate(haf, n->_layers_activations[o][_r-1][i]) * sum;
    }    
  }

}

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::hBackPropagateError(Node* n) {

  int k = _s-1;

  std::vector<float> targets = n->target();
  std::vector<float> outputs(n->_h_layers_activations[_s-1],
			     n->_h_layers_activations[_s-1]+_lnunits[_r+k]);
  require(targets.size() == outputs.size(), "output dim. error");

  /*
   * apply softmax in case of a multi-class (N>2) classification problem
   */
  if(_problem & MULTICLASS) {
    float max = -FLT_MAX;
    for(uint i=0; i<outputs.size(); ++i)
      if(max < outputs[i]) max = outputs[i];
      
    float norm_factor = 0.0;
    for(uint j=0; j<outputs.size(); ++j) {
      outputs[j] = exp(outputs[j] - max);
      norm_factor += outputs[j];
    }
    for(uint j=0; j<outputs.size(); ++j)
      outputs[j] /= norm_factor;
  }

  /*
   * compute errors at the output layer and gradient of
   * the weights from the previous to the output layer
   */
  memset(_delta_h_layers[k], 0, (_lnunits[_r+k])*sizeof(double));
  
  for(int j=0; j<_lnunits[_r+k]; ++j) {
    _delta_h_layers[k][j] =
      (_problem & ~(BINARYCLASS | MULTICLASS)?derivate(oaf, outputs[j]):1.0) *
      (targets[j] - outputs[j]);
      
    if(k>0) {
      /*
       * node output network has hidden layers:
       * gradient at output units depends on delta and 
       * activation at previous (hidden) layer
       */
      for(int i=0; i<_lnunits[_r+k-1]; i++)
	_h_layers_gradient_w[k][i][j] -= 
	  _delta_h_layers[k][j] * n->_h_layers_activations[k-1][i];

      _h_layers_gradient_w[k][_lnunits[_r+k-1]][j] -= 
	_delta_h_layers[k][j];

    } else {
      /*
       * node output network has just one output layer:
       * activations to compute the gradient come from the activation of the output units
       * of the different state transition networks at the node and from its input
       */
      for(int o=0; o<_norient; ++o) {
	for(int i=o*_m; i<(o+1)*_m; ++i)
	   _h_layers_gradient_w[0][i][j] -=
	     _delta_h_layers[k][j] * n->_layers_activations[o][_r-1][i-o*_m];
      }
      
      for(int i=_norient*_m; i<_norient*_m + _n; i++)
	_h_layers_gradient_w[0][i][j] -=
	   _delta_h_layers[k][j] * n->_encodedInput[i-_norient*_m];

      _h_layers_gradient_w[0][_norient*_m+_n][j] -= _delta_h_layers[k][j];
    }
  }

  /*
   * backpropagate error and compute gradient of the weights in the previous layers
   */
  for(--k; k>=0; k--) {
    memset(_delta_h_layers[k], 0, (_lnunits[_r+k])*sizeof(double));

    for(int i=0; i<_lnunits[_r+k]; i++) {
      double sum = 0.0;
      for(int j=0; j<_lnunits[_r+k+1]; j++) {
	sum += _h_layers_w[k+1][i][j] * _delta_h_layers[k+1][j];
      }
      _delta_h_layers[k][i] = 
	derivate(haf, n->_h_layers_activations[k][i]) * sum;
    }

    if(k>0) {
      /*
       * this is a hidden layer but there are more:
       * the activations for the delta rule come from
       * the node unit activations from the previous layer
       */
      for(int j=0; j<_lnunits[_r+k]; j++) {
	for(int i=0; i<_lnunits[_r+k-1]; i++) {
	  _h_layers_gradient_w[k][i][j] -= 
	    _delta_h_layers[k][j] * n->_h_layers_activations[k-1][i];
	}
	_h_layers_gradient_w[k][_lnunits[_r+k-1]][j] -=  _delta_h_layers[k][j];
      }
    } else {
      /*
       * this is the last hidden layer before the inputs whose activations come 
       * from the output units of the different state transition networks and 
       * from the node input
       */
      for(int j=0; j<_lnunits[_r+k]; j++) {
	for(int o=0; o<_norient; ++o) {
	  for(int i=o*_m; i<(o+1)*_m; ++i)
	    _h_layers_gradient_w[0][i][j] -=
	      _delta_h_layers[k][j] * n->_layers_activations[o][_r-1][i-o*_m];
	}
      
	for(int i=_norient*_m; i<_norient*_m + _n; i++)
	  _h_layers_gradient_w[0][i][j] -=
	    _delta_h_layers[k][j] * n->_encodedInput[i-_norient*_m];

	_h_layers_gradient_w[0][_norient*_m+_n][j] -= _delta_h_layers[k][j];
      }
    }
  }

  /*
   * calculate the error on input layer and 
   * redistribute on representation layers of current node.
   */
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<_m; i++) {
      // could probably optimise a little by removing the outer loop
      // and calculating the sums over all possible orientation together
      double sum = 0.0;
      for(int j=0; j<_lnunits[_r]; j++)
	sum += _h_layers_w[0][o*_m + i][j] * _delta_h_layers[0][j];
      
      n->_delta_lr[o][i] += 
	derivate(haf, n->_layers_activations[o][_r-1][i]) * sum;
    }    
  }

}


// Compute Error for a structure in case of Super-Source trasduction
template<class HA_Function, class OA_Function, class EMP>
  double RecursiveNN<HA_Function, OA_Function, EMP>::computeSSError(Instance* instance) {
  // Assume calling training procedure has just 
  // propagated current pattern with target 'targets'.
  std::vector<float> targets = instance->target();
  std::vector<float> outputs(_g_layers_activations[_s-1],
			     _g_layers_activations[_s-1]+_lnunits[_r+_s-1]);
  require(targets.size() == outputs.size(), "output dim. error");

  /*
   * apply softmax in case of a multi-class (N>2) classification problem
   */
  if(_problem & MULTICLASS) {
    float max = -FLT_MAX;
    for(uint i=0; i<outputs.size(); ++i)
      if(max < outputs[i]) max = outputs[i];
      
    float norm_factor = 0.0;
    for(uint j=0; j<outputs.size(); ++j) {
      outputs[j] = exp(outputs[j] - max);
      norm_factor += outputs[j];
    }
    for(uint j=0; j<outputs.size(); ++j)
      outputs[j] /= norm_factor;
  }

  double error = 0.0;
  for(int j=0; j<_lnunits[_r+_s-1]; j++) {
    // compute error contribution for current pattern
    double t = targets[j];
    double o = outputs[j];

    if(_problem & (BINARYCLASS | MULTICLASS)) {
      // cross-entropy
      if(t) error += t * log(t/o);
      if((_problem & BINARYCLASS) && 1-t) error += (1-t) * log((1-t)/(1-o));
      
    } else if (_problem & REGRESSION) {
      // sum of error squares
      error += (t-o) * (t-o);
    } else
      require(0, "Unknown problem");
  }
  
  return error;
}

// Compute Error for a structure in case of IO-Isomorph structural trasduction
template<class HA_Function, class OA_Function, class EMP>
  double RecursiveNN<HA_Function, OA_Function, EMP>::computeIOSError(Instance* instance) {

  double error = .0;
  for(uint n=0; n<instance->num_nodes(); ++n) {
    Node* node = instance->node(n);

    std::vector<float> targets = node->target();
    std::vector<float> outputs(node->_h_layers_activations[_s-1],
			       node->_h_layers_activations[_s-1]+_lnunits[_r+_s-1]);
    require(targets.size() == outputs.size(), "output dim. error");

    /*
     * apply softmax in case of a multi-class (N>2) classification problem
     */
    if(_problem & MULTICLASS) {
      float max = -FLT_MAX;
      for(uint i=0; i<outputs.size(); ++i)
	if(max < outputs[i]) max = outputs[i];
      
      float norm_factor = 0.0;
      for(uint j=0; j<outputs.size(); ++j) {
	outputs[j] = exp(outputs[j] - max);
	norm_factor += outputs[j];
      }
      for(uint j=0; j<outputs.size(); ++j)
	outputs[j] /= norm_factor;
    }

    for(int j=0; j<_lnunits[_r+_s-1]; j++) {
      // compute error contribution for current pattern
      double t = targets[j];
      double o = outputs[j];

      if(_problem & (BINARYCLASS | MULTICLASS)) {
	// cross-entropy
	if(t) error += t * log(t/o);
	if((_problem & BINARYCLASS) && 1-t) error += (1-t) * log((1-t)/(1-o));
      
      } else if (_problem & REGRESSION) {
	// sum of error squares
	error += (t-o) * (t-o);
      } else
	require(0, "Unknown problem");
    } 
  }

  if(_problem & REGRESSION)
    error /= instance->num_nodes();

  return error;
}

template<class HA_Function, class OA_Function, class EMP>
double RecursiveNN<HA_Function, OA_Function, EMP>::computeWeightsNorm() {
  double norm = .0;

  if(_ss_tr) { // begin with layers of the MLP output function
    for(int i=0; i<_norient*_m + 1; ++i)
      for(int j=0; j<_lnunits[_r]; ++j)
	norm += _g_layers_w[0][i][j] * _g_layers_w[0][i][j];

    for(int k=1; k<_s; ++k)
      for(int i=0; i<_lnunits[_r+k-1]+1; i++)
	for(int j=0; j<_lnunits[_r+k]; j++)
	  norm += _g_layers_w[k][i][j] * _g_layers_w[k][i][j];
  }

  if(_ios_tr) { // proceed with layers of the MLP h map
    for(int i=0; i<_norient*_m + _n + 1; ++i)
      for(int j=0; j<_lnunits[_r]; ++j)
	norm += _h_layers_w[0][i][j] * _h_layers_w[0][i][j];

    for(int k=1; k<_s; ++k)
      for(int i=0; i<_lnunits[_r+k-1]+1; ++i)
	for(int j=0; j<_lnunits[_r+k]; ++j) 
	  norm += _h_layers_w[k][i][j] * _h_layers_w[k][i][j];
  }

  // proceed with state transition networks of the various orientations
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<(_n+_v*_m) + 1; ++i)
      for(int j=0; j<_lnunits[0]; ++j)
	norm += _layers_w[o][0][i][j] * _layers_w[o][0][i][j];

    for(int k=1; k<_r; ++k)
      for(int i=0; i<_lnunits[k-1]+1; ++i)
	for(int j=0; j<_lnunits[k]; ++j)
	  norm += _layers_w[o][k][i][j] * _layers_w[o][k][i][j];
  }
  
  return norm;
}


// Evaluate error and performance of the net over a data set
/* template<class HA_Function, class OA_Function, class EMP> */
/* float RecursiveNN<HA_Function, OA_Function, EMP>:: */
/* evaluatePerformanceOnDataSet(DataSet* ds, bool apply_softmax) { */
/*   float error = 0.0; */
/*   for(DataSet::iterator it=ds->begin(); it!=ds->end(); ++it) { */
/*     propagateStructuredInput(*(it.currenTONodes()),  */
/* 			     *(it.currentDPAGs()),  */
/* 			     *(it.currentDPAGsTopOrd())); */

/*     if(_ios_tr) error += computeIOSTrError(*(it.currenTONodes()), apply_softmax); */
/*     if(_ss_tr) error += computeSSTrError(*(it.currentDPAGsTargets()), apply_softmax); */
/*   } */
/*   return error; */
/* } */

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::adjustWeights(float learning_rate, float momentum_term, float ni) {
  // An assertion to safely update weights...
  // In future put this control at the level of training procedure.
  require(0<=learning_rate && learning_rate<=1, "Learning rate interval assertion failed");
  require(0<=momentum_term && momentum_term<1, "Learning rate interval assertion failed");
  require(0<=ni && ni<1, "Regularization coeff. interval assertion failed");

  // Call templatized strategy minimization procedure update method.
  // The optimization strategy type decides to use or not learning rate and/or momentum term
  _wu_method.updateWeights(this, learning_rate, momentum_term, ni);

  // Finally reset gradient components to restart their computation.
  // This works both for classic gradient descent and stochastic approximations.
  resetGradientComponents();
}

template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::restorePrevWeights() {
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<(_n+_v*_m) + 1; i++)
      memcpy(_layers_w[o][0][i], _prev_layers_w[o][0][i], _lnunits[0] * sizeof(double));
  
    for(int k=1; k<_r; k++)
      for(int i=0; i<_lnunits[k-1]+1; i++)
	memcpy(_layers_w[o][k][i], _prev_layers_w[o][k][i], (_lnunits[k])*sizeof(double));
  }
  
  if(_ss_tr) {
    for(int i=0; i<_norient*_m + 1; i++)
      memcpy(_g_layers_w[0][i], _prev_g_layers_w[0][i], _lnunits[_r] * sizeof(double));
    
    for(int k=1; k<_s; k++)
      for(int i=0; i<_lnunits[_r+k-1]+1; i++)
	memcpy(_g_layers_w[k][i], _prev_g_layers_w[k][i], (_lnunits[_r+k])*sizeof(double));
  }

  if(_ios_tr) {
    for(int i=0; i<_norient*_m + _n + 1; i++)
      memcpy(_h_layers_w[0][i], _prev_h_layers_w[0][i], _lnunits[_r] * sizeof(double));
    
    for(int k=1; k<_s; k++)
      for(int i=0; i<_lnunits[_r+k-1]+1; i++)
	memcpy(_h_layers_w[k][i], _prev_h_layers_w[k][i], (_lnunits[_r+k])*sizeof(double));
  }
}

/*** Save network parameters to file ***/
template<class HA_Function, class OA_Function, class EMP>
void RecursiveNN<HA_Function, OA_Function, EMP>::saveParameters(const char* network_filename) {
  std::ofstream os(network_filename);
  assure(os, network_filename);

  // First output processing flags
  os << _norient << ' ' <<_ios_tr << ' ' << _ss_tr << ' ' << endl;
  // Output node input dimension and max outdegree
  // with which the network was trained
  os << _n << ' ' << _v << endl;
  // Ouput to file values of r and s.
  os << _r << ' ' << _s << endl;

  // N.B. All previous values must be synchronized
  // with global Options parameters to be sure to 
  // have consistent values across different objects (RNN, Node, DataSet...)

  // Then the vector of number of units per layer
  for(std::vector<int>::const_iterator it=_lnunits.begin();
      it!=_lnunits.end(); ++it)
    os << *it << ' ';
  os << endl << endl;

  os.precision(Options::instance()->precision());
  os.setf(std::ios::scientific);
  /************************/

  // Then output network weights to file
  // First the weights for each state transition function
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      for(int j=0; j<_lnunits[0]; j++)
	os << _layers_w[o][0][i][j] << ' ';
      
      os << endl;
    }
    os << endl;

    for(int k=1; k<_r; k++) {
      for(int i=0; i<_lnunits[k-1]+1; i++) {
	for(int j=0; j<_lnunits[k]; j++)
	  os << _layers_w[o][k][i][j] << ' ';
	
	os << endl;
      }
      os << endl;
    }
    os << endl;
  }
  
  // Warning!! the following order of writing is important
  // for the constructor that read network from file.
  if(_ss_tr) {
    // Now output weights for g output function layers
    for(int i=0; i<_norient*_m + 1; i++) {
      for(int j=0; j<_lnunits[_r]; j++)
	os << _g_layers_w[0][i][j] << ' ';

      os << endl;
    }
    os << endl;

    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++)
	  os << _g_layers_w[k][i][j] << ' ';

	os << endl;
      }
      os << endl;
    }
    os << endl;
  }

  if(_ios_tr) {
    // Now output weights for h map layers
    for(int i=0; i<_norient*_m + _n + 1; i++) {
      for(int j=0; j<_lnunits[_r]; j++)
	os << _h_layers_w[0][i][j] << ' ';

      os << endl;
    }
    os << endl;

    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++)
	  os << _h_layers_w[k][i][j] << " ";

	os << endl;
      }
      os << endl;
    }
    os << endl;
  }
  
}


#endif // RECURSIVE_NN_H
