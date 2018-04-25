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

#ifndef _PERFORMANCE_H_
#define _PERFORMANCE_H_

#include "StructuredDomain.h"
#include "Instance.h"

#include <string>
#include <iostream>

class Performance {
 public:
  Performance();
  virtual ~Performance() {}
  virtual void reset() = 0;
  virtual void update(Instance*) = 0;
  virtual void print(std::ostream& = std::cout) = 0;
  
  class BadPerformanceCreation: public std::logic_error {
    public:
      BadPerformanceCreation(std::string msg): logic_error(msg) {}
   };

  // factory methods
  static Performance* factory(const Problem&)
    throw(BadPerformanceCreation);

  friend std::ostream& operator<<(std::ostream& os, Performance* p) {
    p->print(os);
    return os;
  }
  
 protected:
  Problem _problem;
  bool _ss_tr, _ios_tr;
 
};

class ClassificationPerformance: public Performance {
 public:
  ~ClassificationPerformance();
  void reset();
  void update(Instance*);
  void print(std::ostream&);
  
 private:
  ClassificationPerformance(int);
  friend class Performance;

  int _nc; // number of classes
  int** _cm; // confusion matrix

  int num_class_instances(int) const;
  int num_class_errors(int) const;
  int num_total_errors() const;
};

// TODO
// report error, sum of squared residuals, correlation
class RegressionPerformance: public Performance {
 public:
  void reset() {}
  void update(Instance*) {}
  void print(std::ostream&) {}
  
 private:
  RegressionPerformance() {}
  friend class Performance;

};

#endif // _PERFORMANCE_H_
