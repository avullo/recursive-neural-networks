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

#ifndef _INSTANCE_PARSER_H_
#define _INSTANCE_PARSER_H_

#include "Instance.h"
#include <fstream>

class InstanceParser {
  Domain _domain;
  Transduction _transduction;
  bool _supervised;

  std::string _id;
  unsigned int _num_nodes, _input_dim, _output_dim;

  Instance* _instance;
  
  std::istream& read_header(std::istream&);
  std::istream& read_node_io(std::istream&);

  std::istream& read_skeleton(std::istream&);
  std::istream& read_sequence(std::istream&);
  std::istream& read_linear_chain(std::istream&);
  std::istream& read_doag(std::istream&);
  std::istream& read_ugraph(std::istream&);
  std::istream& read_grid2d(std::istream&);

  // prevent assignment and copy construction
  InstanceParser(const InstanceParser&);
  InstanceParser& operator=(const InstanceParser&);
  
 public:
  InstanceParser(bool = true /* supervised (i,e. labelled) */);
  Instance* read(std::istream&);
  // TODO: write method
  void write(std::ostream&) {}
};

#endif // _INSTANCE_PARSER_H_
