/*
 * Recursive Neural Networks: neural networks for data structures 
 *
 * Copyright (C) 2018 Alessandro Vullo 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/


/*
  Templatized Strategy Pattern implementation
  of different Error Minimization Procedures.
*/

#ifndef _ERROR_MINIMIZATION_PROCEDURE_H
#define _ERROR_MINIMIZATION_PROCEDURE_H

#include <vector>
#include <iostream>
using std::cout;
using std::endl;

/* Simple Gradient Descent */
class GradientDescent {
  // Simply update networks weights in the opposite direction 
  // of the current gradient stored in the net and with a 
  // distance defined by learning rate.
  bool _ss_tr, _ios_tr;
  int _norient, _n, _v, _m, _r, _s;
  std::vector<int> _lnunits;

  public:
  template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
    void setInternals(RNN<T1, T2, T3>* const rnn);

  template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
    void updateWeights(RNN<T1, T2, T3>* const rnn, double, double = 0.0);
};

template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
  void GradientDescent::setInternals(RNN<T1, T2, T3>* const rnn) {
    // Simply synchronize private member with those of recursive network
    _norient = rnn->_norient;
    _ss_tr = rnn->_ss_tr;
    _ios_tr = rnn->_ios_tr;
    _n = rnn->_n, _v = rnn->_v, _m = rnn->_m, _r = rnn->_r, _s = rnn->_s;
    _lnunits = rnn->_lnunits;
}

template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN> 
void GradientDescent::updateWeights(RNN<T1, T2, T3>* const rnn, double _learning_rate, double) {
  // rnn (the recursive network) store gradient components, so to obtain
  // weight update rule change the sign of these components and multiply
  // by the learning rate
  
  if(_ss_tr) {
    // begin with layers of the MLP output function
    for(int i=0; i<_norient*_m; i++) {
      for(int j=0; j<_lnunits[_r]; j++) {
	// Save parameters in case we make a bad move.
	rnn->_prev_g_layers_w[0][i][j] = rnn->_g_layers_w[0][i][j];
	rnn->_g_layers_w[0][i][j] += -_learning_rate * rnn->_g_layers_gradient_w[0][i][j];
      }
    }
    // this could go inside the previous loop
    for(int j=0; j<_lnunits[_r]; j++) {
      rnn->_prev_g_layers_w[0][_norient*_m][j] = rnn->_g_layers_w[0][_norient*_m][j];
      rnn->_g_layers_w[0][_norient*_m][j] += -_learning_rate * rnn->_g_layers_gradient_w[0][_norient*_m][j];
    }
    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++) {
	  rnn->_prev_g_layers_w[k][i][j] = rnn->_g_layers_w[k][i][j];
	  rnn->_g_layers_w[k][i][j] += -_learning_rate * rnn->_g_layers_gradient_w[k][i][j];
	}
      }
    }
  }

  if(_ios_tr) {
    // proceed with layers of the h map
    for(int i=0; i<_norient*_m+_n; i++) {
      for(int j=0; j<_lnunits[_r]; j++) {
	rnn->_prev_h_layers_w[0][i][j] = rnn->_h_layers_w[0][i][j];
	rnn->_h_layers_w[0][i][j] += -_learning_rate * rnn->_h_layers_gradient_w[0][i][j];
      }
    }
    // this could inside the previous loop
    for(int j=0; j<_lnunits[_r]; j++) {
      rnn->_prev_h_layers_w[0][_norient*_m + _n][j] = rnn->_h_layers_w[0][_norient*_m + _n][j];
      rnn->_h_layers_w[0][_norient*_m + _n][j] += -_learning_rate * rnn->_h_layers_gradient_w[0][_norient*_m + _n][j];
    }
    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++) {
	  rnn->_prev_h_layers_w[k][i][j] = rnn->_h_layers_w[k][i][j];
	  rnn->_h_layers_w[k][i][j] += -_learning_rate * rnn->_h_layers_gradient_w[k][i][j];
	}
      }
    }
  }

  // proceed with the folding layers
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      for(int j=0; j<_lnunits[0]; j++) {
	rnn->_prev_layers_w[o][0][i][j] = rnn->_layers_w[o][0][i][j];
	rnn->_layers_w[o][0][i][j] += -_learning_rate * rnn->_layers_gradient_w[o][0][i][j];
      }
    }
    
    for(int k=1; k<_r; k++) {
      for(int i=0; i<_lnunits[k-1]+1; i++) {
	for(int j=0; j<_lnunits[k]; j++) {
	  rnn->_prev_layers_w[o][k][i][j] = rnn->_layers_w[o][k][i][j];
	  rnn->_layers_w[o][k][i][j] += -_learning_rate * rnn->_layers_gradient_w[o][k][i][j];
	}
      }
    }
  }
  
}

/* Gradient Descent with momentum */
class MGradientDescent {
  // Store previuos step weights delta values and update
  // weights with net current gradient and these values.
  double**** _layers_old_deltas_w;
  double*** _g_layers_old_deltas_w;
  double*** _h_layers_old_deltas_w;

  bool _ss_tr, _ios_tr;
  int _norient, _n, _v, _m, _r, _s;
  std::vector<int> _lnunits;

  void allocFoldingPart(double**** layers_w) {
    (*layers_w) = new double**[_r];
  
    // Allocate weights and gradient components matrixes
    // for f folding part. Connections from input layer
    // have special dimensions.
    (*layers_w)[0] = new double*[(_n + _v * _m) + 1];

    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      (*layers_w)[0][i] = new double[_lnunits[0]];

      // Reset gradient for corresponding weights
      memset((*layers_w)[0][i], 0, _lnunits[0] * sizeof(double));
    }
  
    // Allocate f weights&gradient matrixes for f folding part.
    for(int k=1; k<_r; k++) {
      // Allocate space for weight&delta matrix between layer i-1 and i.
      // Automatically include space for threshold unit in layer i-1
      (*layers_w)[k] = new double*[_lnunits[k-1] + 1];

      for(int i=0; i<_lnunits[k-1]+1; i++) {
	(*layers_w)[k][i] = new double[_lnunits[k]];

	// Reset f gradient components for corresponding weights
	memset((*layers_w)[k][i], 0, (_lnunits[k])*sizeof(double));
      } 
    }
  }

  void allocSSPart() {
    _g_layers_old_deltas_w = new double**[_s];

    // Allocate weights and gradient components matrixes
    // for g transforming part. Connections from input layers
    // have special dimensions.
    _g_layers_old_deltas_w[0] = new double*[_norient*_m + 1];

    for(int i=0; i<_norient*_m + 1; i++) {
      _g_layers_old_deltas_w[0][i] = new double[_lnunits[_r]];

      // Reset gradient for corresponding weights
      memset(_g_layers_old_deltas_w[0][i], 0, _lnunits[_r] * sizeof(double));
    }
  
    // Allocate weights&gradient matrixes for g transforming part
    for(int k=1; k<_s; k++) {
      // Allocate space for weight&gradient matrixes between layer i-1 and i.
      // Automatically include space for threshold unit in layer i-1
      _g_layers_old_deltas_w[k] = new double*[_lnunits[_r+k-1] + 1];

      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	_g_layers_old_deltas_w[k][i] = new double[_lnunits[_r+k]];

	// Reset g gradient components for corresponding weights
	memset(_g_layers_old_deltas_w[k][i], 0, (_lnunits[_r+k])*sizeof(double));
      }
    }
  }

  void allocIOSPart() {
    _h_layers_old_deltas_w = new double**[_s];
    
    // Allocate weights and gradient components matrixes
    // for h map. Connections from input layers have special dimensions.
    _h_layers_old_deltas_w[0] = new double*[_norient*_m + _n + 1];

    for(int i=0; i<_norient*_m + _n + 1; i++) {
      _h_layers_old_deltas_w[0][i] = new double[_lnunits[_r]];

      // Reset gradient for corresponding weights
      memset(_h_layers_old_deltas_w[0][i], 0, _lnunits[_r] * sizeof(double));
    }
  
    // Allocate weights&gradient matrixes for h map
    for(int k=1; k<_s; k++) {
      // Allocate space for weight&gradient matrixes between layer i-1 and i.
      // Automatically include space for threshold unit in layer i-1
      _h_layers_old_deltas_w[k] = new double*[_lnunits[_r+k-1] + 1];

      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	_h_layers_old_deltas_w[k][i] = new double[_lnunits[_r+k]];

	// Reset h gradient components for corresponding weights
	memset(_h_layers_old_deltas_w[k][i], 0, (_lnunits[_r+k])*sizeof(double));
      }
    }
  }

  void deallocFoldingPart(double**** layers_w) {
    if((*layers_w)[0]) {
      for(int i=0; i<(_n+_v*_m) + 1; i++) {
	if((*layers_w)[0][i])
	  delete[] (*layers_w)[0][i];
	(*layers_w)[0][i] = 0;
      }
      delete[] (*layers_w)[0];
      (*layers_w)[0] = 0;
    }

    for(int k=1; k<_r; k++) {
      if((*layers_w)[k]) {
	for(int i=0; i<_lnunits[k-1]+1; i++) {
	  if((*layers_w)[k][i])
	    delete[] (*layers_w)[k][i];
	  (*layers_w)[k][i] = 0;
	}
	delete[] (*layers_w)[k];
	(*layers_w)[k] = 0;
      }
    
    }
  
    delete[] (*layers_w);
    (*layers_w) = 0;
  }

  void deallocSSPart() {
    if(_g_layers_old_deltas_w[0]) {
      for(int i=0; i<_norient*_m + 1; i++) {
	if(_g_layers_old_deltas_w[0][i])
	  delete[] _g_layers_old_deltas_w[0][i];
	_g_layers_old_deltas_w[0][i] = 0;
      }
      delete[] _g_layers_old_deltas_w[0];
      _g_layers_old_deltas_w[0] = 0;
    }

    for(int k=1; k<_s; k++) {
      if(_g_layers_old_deltas_w[k]) {
	for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	  if(_g_layers_old_deltas_w[k][i])
	    delete[] _g_layers_old_deltas_w[k][i];
	  _g_layers_old_deltas_w[k][i] = 0;
	}
	delete[] _g_layers_old_deltas_w[k];
	_g_layers_old_deltas_w[k] = 0;
      }
    }
  
    delete[] _g_layers_old_deltas_w;
    _g_layers_old_deltas_w = 0;
  }

  void deallocIOSPart() {
    if(_h_layers_old_deltas_w[0]) {
      for(int i=0; i<_norient*_m + _n + 1; i++) {
	if(_h_layers_old_deltas_w[0][i])
	  delete[] _h_layers_old_deltas_w[0][i];
	_h_layers_old_deltas_w[0][i] = 0;
      }
      delete[] _h_layers_old_deltas_w[0];
      _h_layers_old_deltas_w[0] = 0;
    }

    for(int k=1; k<_s; k++) {
      if(_h_layers_old_deltas_w[k]) {
	for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	  if(_h_layers_old_deltas_w[k][i])
	    delete[] _h_layers_old_deltas_w[k][i];
	  _h_layers_old_deltas_w[k][i] = 0;
	}
	delete[] _h_layers_old_deltas_w[k];
	_h_layers_old_deltas_w[k] = 0;
      }
    }
  
    delete[] _h_layers_old_deltas_w;
    _h_layers_old_deltas_w = 0;
  }
  
 public:
  ~MGradientDescent();

  template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
    void setInternals(RNN<T1, T2, T3>* const rnn);

  template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
    void updateWeights(RNN<T1, T2, T3>* const rnn, float = .0, float = .0, float = .0);
};

MGradientDescent::~MGradientDescent() {
  for(int i=0; i<_norient; ++i)
    deallocFoldingPart(&(_layers_old_deltas_w[i]));
    
  if(_ss_tr) deallocSSPart();
  if(_ios_tr) deallocIOSPart();
}

template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
  void MGradientDescent::setInternals(RNN<T1, T2, T3>* const rnn) {
  // Assume rnn constructor has initialized 
  // required dimension quantities.
  _norient = rnn->_norient;
  _ss_tr = rnn->_ss_tr; _ios_tr = rnn->_ios_tr;
  _n = rnn->_n, _v = rnn->_v, _m = rnn->_m, _r = rnn->_r, _s = rnn->_s;
  _lnunits = rnn->_lnunits;

  _layers_old_deltas_w = new double***[_norient];
  for(int i=0; i<_norient; ++i)
    allocFoldingPart(&(_layers_old_deltas_w[i]));
  
  // Finally allocate space for g and or h gradient structures
  if(_ss_tr) allocSSPart();
  if(_ios_tr) allocIOSPart();
}

template<typename T1, typename T2, typename T3, template<typename, typename, typename> class RNN>
  void MGradientDescent::updateWeights(RNN<T1, T2, T3>* const rnn, float _learning_rate, float momentum_term, float ni) {
  // rnn (the recursive network) store gradient components, so to obtain
  // weight update rule change the sign of this components, multiply
  // by the learning rate and add multiplication of momentum_term
  // with old weights deltas.
    
  float new_delta_w = .0;
  if(_ss_tr) { // begin with layers of the MLP output function
    for(int i=0; i<_norient*_m + 1; i++) {
      for(int j=0; j<_lnunits[_r]; j++) {
	rnn->_prev_g_layers_w[0][i][j] = rnn->_g_layers_w[0][i][j];
	new_delta_w = 
	  -_learning_rate * rnn->_g_layers_gradient_w[0][i][j] +
	  (momentum_term * _g_layers_old_deltas_w[0][i][j]) -
	  (ni * rnn->_g_layers_w[0][i][j]);
	
	rnn->_g_layers_w[0][i][j] += new_delta_w;
	_g_layers_old_deltas_w[0][i][j] = new_delta_w;
      }
    }
    
    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++) {
	  rnn->_prev_g_layers_w[k][i][j] = rnn->_g_layers_w[k][i][j];
	  new_delta_w =  
	    -_learning_rate * rnn->_g_layers_gradient_w[k][i][j] +
	    (momentum_term * _g_layers_old_deltas_w[k][i][j]) -
	    (ni * rnn->_g_layers_w[k][i][j]);;
	  
	  rnn->_g_layers_w[k][i][j] += new_delta_w;
	  _g_layers_old_deltas_w[k][i][j] = new_delta_w;
	}
      }
    }
  }

  if(_ios_tr) { // proceed with layers of the MLP h map
    new_delta_w = 0.0;
    
    for(int i=0; i<_norient*_m + _n + 1; i++) {
      for(int j=0; j<_lnunits[_r]; j++) {
	rnn->_prev_h_layers_w[0][i][j] = rnn->_h_layers_w[0][i][j];
	new_delta_w =  
	  -_learning_rate * rnn->_h_layers_gradient_w[0][i][j] +
	  (momentum_term * _h_layers_old_deltas_w[0][i][j]) -
	  (ni * rnn->_h_layers_w[0][i][j]);
	
	rnn->_h_layers_w[0][i][j] += new_delta_w;
	_h_layers_old_deltas_w[0][i][j] = new_delta_w;
      }
    }

    for(int k=1; k<_s; k++) {
      for(int i=0; i<_lnunits[_r+k-1]+1; i++) {
	for(int j=0; j<_lnunits[_r+k]; j++) {
	  rnn->_prev_h_layers_w[k][i][j] = rnn->_h_layers_w[k][i][j];
	  new_delta_w =  
	    -_learning_rate * rnn->_h_layers_gradient_w[k][i][j] +
	    (momentum_term * _h_layers_old_deltas_w[k][i][j]) -
	    (ni * rnn->_h_layers_w[k][i][j]);
	  
	  rnn->_h_layers_w[k][i][j] += new_delta_w;
	  _h_layers_old_deltas_w[k][i][j] = new_delta_w;
	}
      }
    }
  }

  // proceed with the folding layers
  for(int o=0; o<_norient; ++o) {
    for(int i=0; i<(_n+_v*_m) + 1; i++) {
      for(int j=0; j<_lnunits[0]; j++) {
	rnn->_prev_layers_w[o][0][i][j] = rnn->_layers_w[o][0][i][j];
	new_delta_w =  
	  -_learning_rate * rnn->_layers_gradient_w[o][0][i][j] +
	  (momentum_term * _layers_old_deltas_w[o][0][i][j]) -
	  (ni * rnn->_layers_w[o][0][i][j]);
	
	rnn->_layers_w[o][0][i][j] += new_delta_w;
	_layers_old_deltas_w[o][0][i][j] = new_delta_w;
	
      }
    }
    
    for(int k=1; k<_r; k++) {
      for(int i=0; i<_lnunits[k-1]+1; i++) {
	for(int j=0; j<_lnunits[k]; j++) {
	  rnn->_prev_layers_w[o][k][i][j] = rnn->_layers_w[o][k][i][j];
	  new_delta_w =  
	    -_learning_rate * rnn->_layers_gradient_w[o][k][i][j] +
	    (momentum_term * _layers_old_deltas_w[o][k][i][j]) -
	    (ni * rnn->_layers_w[o][k][i][j]);
	  
	  rnn->_layers_w[o][k][i][j] += new_delta_w;
	  _layers_old_deltas_w[o][k][i][j] = new_delta_w;

	}
      }
    }
  }
  
}

#endif // _ERROR_MINIMIZATION_PROCEDURE_H
