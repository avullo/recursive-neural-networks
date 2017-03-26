#include "General.h"
#include "InstanceParser.h"
using namespace std;

InstanceParser::InstanceParser(bool supervised):
  _domain(Options::instance()->domain()), _transduction(Options::instance()->transduction()), _supervised(supervised) {

  _input_dim = Options::instance()->input_dim();
  _output_dim = Options::instance()->output_dim();
  _num_nodes = -1;
}

Instance* InstanceParser::read(istream& is) {
  // no need to deallocate, whoever calls me get
  // responsibility of the allocated memory for the instance
  _instance = new Instance(_domain, _transduction, _supervised);
 
  read_header(is);
  read_node_io(is);
  read_skeleton(is);

  return _instance;
}

// TODO: not sure I don't have to read target when not supervised
// read header (various) i/o dimensions
istream& InstanceParser::read_header(istream& is) {
  string id;
  is >> id >> _num_nodes;
  assert(id.size());  
  assert(_num_nodes > 0);
  _instance->id(id);
  
  if(_transduction == SUPER_SOURCE && _supervised) {
    vector<float> graph_target(_output_dim, .0);
    for(uint i=0; i<_output_dim; ++i)
      is >> graph_target[i];
    _instance->load_target(graph_target);
  }
  
  return is;
}

istream& InstanceParser::read_node_io(istream& is) {
  for(uint i=0; i<_num_nodes; ++i) {
    Node* n = new Node;
    
    vector<float> node_input(_input_dim, .0);
    for(uint j=0; j<_input_dim; ++j)
      is >> node_input[j];
    n->load_input(node_input);
    
    if(_transduction == IO_ISOMORPH && _supervised) {
      vector<float> node_target(_output_dim, .0);
      for(uint j=0; j<_output_dim; ++j)
	is >> node_target[j];
      n->load_target(node_target);
    }
    
    _instance->_nodes.push_back(n);
  }

  return is;
}

istream& InstanceParser::read_skeleton(istream& is) {
  // dependening on the domain, read or build node connectivity
  switch(_domain) {
  case SEQUENCE: read_sequence(is); break;
  case LINEARCHAIN: read_linear_chain(is); break;
  case DOAG: read_doag(is); break;
  case UG: read_ugraph(is); break;
  case GRID2D: read_grid2d(is); break;
  default: throw Instance::BadInstanceCreation("Unknown domain");
  }

  return is;
}

istream& InstanceParser::read_sequence(std::istream& is) {
  Instance::Skeleton* skel = new Instance::Skeleton(_domain);
  DPAG* sequence = new DPAG(_num_nodes);
  
  // make the connections go from right to left
  // to emulate RNN unfolding from left to right,
  // i.e. reverse topological sort
  for(uint i=_num_nodes-1; i>0; --i)
    boost::add_edge(i, i-1, EdgeProperty(0), *sequence);
  
  skel->orientation(0, sequence);
  _instance->skeleton(skel);

  return is;
}

istream& InstanceParser::read_linear_chain(std::istream& is) {
  // skeleton is composed of two sequences, one right to left
  // and the other from left to right
  Instance::Skeleton* skel = new Instance::Skeleton(_domain);

  DPAG* rlseq = new DPAG(_num_nodes);
  for(uint i=_num_nodes-1; i>0; --i)
    boost::add_edge(i, i-1, EdgeProperty(0), *rlseq);
  skel->orientation(0, rlseq);

  DPAG* lrseq = new DPAG(_num_nodes);
  for(uint i=0; i<_num_nodes-1; ++i)
    boost::add_edge(i, i+1, EdgeProperty(0), *lrseq);
  skel->orientation(1, lrseq);

  _instance->skeleton(skel);
  
  return is;
}

istream& InstanceParser::read_doag(std::istream& is) {
  Instance::Skeleton* skel = new Instance::Skeleton(_domain);

  DPAG* doag = new DPAG(_num_nodes);
  string line;
  for(uint i=0; i<_num_nodes; ++i) {
    getline(is, line);
    istringstream iss(line);
    int v;
    iss >> v;
    int target;
    int eindex = 0;
    while(iss >> target)
      boost::add_edge(v, target, EdgeProperty(eindex++), *doag);
  }

  skel->orientation(0, doag);
  _instance->skeleton(skel);

  return is;
}

istream& InstanceParser::read_ugraph(std::istream& is) { return is; }

istream& InstanceParser::read_grid2d(std::istream& is) { return is; }
