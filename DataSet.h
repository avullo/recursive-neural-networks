#ifndef DATA_SET_H
#define DATA_SET_H

#include "Instance.h"
#include <vector>

class DataSet: public std::vector<Instance*> {
  
 public:
  DataSet() {}
  DataSet(const char*);
  ~DataSet();

  void add(Instance*);
  void shuffle();
};

#endif // DATA_SET_H
