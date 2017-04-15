#ifndef _ACTIVATION_FUNCTIONS_H
#define _ACTIVATION_FUNCTIONS_H

#include <cmath>

/***
    Callback inlining technique based on STL-style function objects.
    The aim is to create parameterized evaluation and derivation
    function of non linear (sigmoid, tanh) and linear activation
    functions commonly adopted in Neural Networks programming practice.
***/

class Sigmoid {
 public:
  double operator()(double x) {
    return (1.0 / (1.0 + exp(-x)));
  }

  double deriv(double x) {
    return x * (1.0 - x);
  }
};

class TanH {
 public:
  double operator()(double x) {
    return tanh(x);
  }

  double deriv(double x) {
    return (1.0 - x * x);
  }
};

class Linear {
 public:
  double operator()(double x) {
    return x;
  }

  double deriv(double x) {
    return 1.0;
  }
};

class LinearSaturated {
 public:
  double operator()(double x) {
    if(x >= 1)
      return 1;
    else if(x <= -1)
      return 0;
    else
      return (x + 1) / 2;
  }

  double deriv(double x) {
    if(x > 1 || x < -1)
      return 0;
    else
      return .5;
  }
};


// We can now define templatized functions
// to evaluate and derivate the unit activation functions.

template<class T_function>
double evaluate(T_function f, double x) {
  return f(x);
}

template<class T_function>
double derivate(T_function f, double x) {
  return f.deriv(x);
}


/***
    First attempt to create parameterized activation function,
    based on polymorphism. 
    Rejected because virtual function dispatch will cause
    poor performance.
***/


/*
class ActivationFunction {
 public:
  virtual double function(double x) = 0;
  virtual double derivative_of_function(double x) = 0;

  double evaluate(double x) {
    return function(x);
  }

  double derivate(double x) {
    return derivative_of_function(x);
  }
};

class Sigmoid: public ActivationFunction {
 public:
  virtual double function(double x) {
    return (1.0 / (1.0 + exp(-x)));
  }

  virtual double derivative_of_function(double x) {
    return x * (1.0 - x);
  }
};

class Tanh: public ActivationFunction {
 public:
  virtual double function(double x) {
    return tanh(x);
  }

  virtual double derivative_of_function(double x) {
    return (1.0 - x * x);
  }
};

class Linear: public ActivationFunction {
 public:
  virtual double function(double x) {
    return x;
  }

  virtual double derivative_of_function(double x) {
    return 1.0;
  }
};
*/

#endif // _ACTIVATION_FUNCTIONS_H
