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
