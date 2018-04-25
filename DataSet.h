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
