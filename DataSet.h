#ifndef DATA_SET_H
#define DATA_SET_H

#include "Instance.h"
#include <vector>

class DataSet: public std::vector<Instance*> {
  // would include a map from instance names to positions
  // to be able to create partition on demand

  // whether the collection owns the instance pointers
  // so as the destructor can proceed
  bool _own;
  int _nnodes; // total number of nodes in the dataset

 public:
  // flag signal pointer ownership
 DataSet(bool own = true): _own(own), _nnodes(0) {}
  DataSet(const char*, bool = true);
  ~DataSet();

  void add(Instance*);
  void shuffle();
  int num_nodes() const { return _nnodes; }
};

#endif // DATA_SET_H
