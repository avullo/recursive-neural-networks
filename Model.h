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

#ifndef _MODEL_H_
#define _MODEL_H_

#include "Instance.h"
#include "RecursiveNN.h"

/*
 * Represent a RNN model
 */

class Model {
 public:

  Model() {}
  virtual ~Model() {}

  virtual void read(const char*) = 0;
  virtual void write(const char*) = 0;

  virtual void predict(Instance*, std::ostream& = cout);
  
 protected:

  bool _ios_tr, _ss_tr;
};

class BinaryClassModel: public Model {
  RecursiveNN<TanH, Sigmoid, MGradientDescent>* _rnn;

 public:
  ~BinaryClassModel();

  void read(const char*);
  void write(const char*);

  void predict(Instance*, std::ostream& = cout);
};

class MultiClassModel: public Model {
  RecursiveNN<TanH, Linear, MGradientDescent>* _rnn;

 public:
  ~MultiClassModel();

  void read(const char*);
  void write(const char*);

  void predict(Instance*, std::ostream& = cout);
};

class RegressionModel: public Model {
  RecursiveNN<TanH, Sigmoid, MGradientDescent>* _rnn;

  void read(const char*);
  void write(const char*);

  void predict(Instance*, std::ostream& = cout);
};


#endif // _MODEL_H_
