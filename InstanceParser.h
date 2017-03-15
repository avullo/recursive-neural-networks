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
  
 public:
  InstanceParser(bool = true /* supervised (i,e. labelled) */);
  Instance* read(std::istream&);
  // TODO: write method
  void write(std::ostream&) {}
};

#endif // _INSTANCE_PARSER_H_
