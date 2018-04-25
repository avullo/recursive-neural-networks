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

#include "Instance.h"

#include <cassert>
#include <algorithm>
using namespace std;

Instance::Skeleton::Skeleton(Domain domain): _i(-1), _o(-1), _norient(num_orientations(domain)) {
  assert(_norient > 0);
  
  _orientations = new DPAG*[_norient];
  _top_orders = new vector<int>[_norient];
  
  for(uint i=0; i<_norient; ++i)
    _orientations[i] = NULL;

}

Instance::Skeleton::~Skeleton() {
  for(uint i=0; i<_norient; ++i)
    delete _orientations[i];
  delete[] _orientations;  
  delete[] _top_orders;
}


void Instance::Skeleton::orientation(uint index, DPAG* dpag) {
  assert(index>=0 && index<_norient);

  int mi = max_indegree(*dpag);
  int mo = max_outdegree(*dpag);
  if(_i < mi) _i = mi;
  if(_o < mo) _o = mo;

  if(_orientations[index] != NULL)
    delete _orientations[index];
  _orientations[index] = dpag;
  
  _top_orders[index] = topological_sort(*(_orientations[index]));
}

DPAG* Instance::orientation(uint index) {
  assert(index>=0 && index<_skel->_norient);
  
  return _skel->_orientations[index];
}

vector<int> Instance::topological_order(uint index) const {
  assert(index>=0 && index<_skel->_norient);

  return _skel->_top_orders[index];
}

void Instance::print(ostream& os) {
  os << "-- " << id() << " --" << endl << endl;
  for(uint i=0; i<num_nodes(); ++i) {
    os << i << '\t';
    Node* n = node(i);
    vector<float> input = n->input();
    vector<float> target = n->target();
    copy(input.begin(), input.end(), ostream_iterator<float>(os, ","));
    os << " | ";
    copy(target.begin(), target.end(), ostream_iterator<float>(os, ","));
    os << endl;
  }

  printDPAG(*orientation(0), os);
  os << endl;
}
