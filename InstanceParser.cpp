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

#include "General.h"
#include "Options.h"
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
  _num_nodes = -1;
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

  // move to the first non-empty line after node i/o
  is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while(is.peek()=='\n')
    is.get();
  
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

istream& InstanceParser::read_ugraph(std::istream& is) {
  Instance::Skeleton* skel = new Instance::Skeleton(_domain);

  // move to the first non-empty line after node i/o
  is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while(is.peek()=='\n')
    is.get();

  DPAG dpag(_num_nodes);
  string line;
  for(uint i=0; i<_num_nodes; ++i) {
    getline(is, line);
    istringstream iss(line);
    int v;
    iss >> v;
    int target;
    int eindex = 0;
    while(iss >> target)
      boost::add_edge(v, target, EdgeProperty(eindex++), dpag);
  }

  VertexId vertex_id = boost::get(boost::vertex_index, dpag);
  vertexIt v_i, v_end;
  outIter out_i, out_end, out_i1;

  map<int, vector<int> > r_edges;
  uint n_vertices = boost::num_vertices(dpag);
  for(uint i=0; i<n_vertices; i++) {
    r_edges[i] = vector<int>();
  }

  DPAG* d_dpag = new DPAG(n_vertices);
  for(boost::tie(v_i, v_end) = boost::vertices(dpag); v_i!=v_end; ++v_i) {
    vector<uint> children;
    for(boost::tie(out_i, out_end)=boost::out_edges(*v_i, dpag);
	out_i!=out_end; ++out_i) {
      children.push_back(boost::target(*out_i, dpag));
      r_edges[boost::target(*out_i, dpag)].push_back(boost::source(*out_i, dpag));
    }
    if(vertex_id[*v_i] != n_vertices-1) {
      int edge_index = 0;
      if(!children.size()) {
	boost::add_edge(vertex_id[*v_i], vertex_id[*v_i]+1, EdgeProperty(edge_index), *d_dpag);
      } else {
	sort(children.begin(), children.end());

	if(children.front() == vertex_id[*v_i]+1)
	  edge_index++; 
	else 
	  boost::add_edge(vertex_id[*v_i], vertex_id[*v_i]+1, EdgeProperty(edge_index++), *d_dpag);
	 
	for(uint i=0; i<children.size(); i++)
	  boost::add_edge(vertex_id[*v_i], children[i], EdgeProperty(edge_index++), *d_dpag);
      }
    }
  }
  
  DPAG* r_dpag = new DPAG(n_vertices);
  for(boost::tie(v_i, v_end) = boost::vertices(dpag); v_i!=v_end; ++v_i) {
    if(!vertex_id[*v_i])
      continue;
    map<int, vector<int> >::iterator it = r_edges.find(vertex_id[*v_i]);
    if(it != r_edges.end()) {
      int edge_index = 0;
      if(!(*it).second.size()) {
	boost::add_edge(vertex_id[*v_i], vertex_id[*v_i]-1, EdgeProperty(edge_index), *r_dpag);
      } else {
	sort((*it).second.begin(), (*it).second.end(), greater<int>());

	if((uint)(*it).second.front() == vertex_id[*v_i]-1)
	  edge_index++; 
	else 
	  boost::add_edge(vertex_id[*v_i], vertex_id[*v_i]-1, EdgeProperty(edge_index++), *r_dpag);
	 
	for(uint i=0; i<(*it).second.size(); i++)
	  boost::add_edge(vertex_id[*v_i], ((*it).second)[i], EdgeProperty(edge_index++), *r_dpag);
      }
    }
  }

  skel->orientation(0, d_dpag);
  skel->orientation(1, r_dpag);
  _instance->skeleton(skel);
  
  return is;
}

istream& InstanceParser::read_grid2d(std::istream& is) {
  Instance::Skeleton* skel = new Instance::Skeleton(_domain);

  // read the number of rows and columns
  int rows, cols;
  is >> rows >> cols;
  assert(rows>0 && cols>0);
  assert(_num_nodes == (uint)rows*cols);

  DPAG* nwse_grid = new DPAG(_num_nodes);
  build_grid("nwse", rows, cols, nwse_grid);
  skel->orientation(0, nwse_grid);

  DPAG* senw_grid = new DPAG(_num_nodes);
  build_grid("senw", rows, cols, senw_grid);
  skel->orientation(1, senw_grid);

  DPAG* nesw_grid = new DPAG(_num_nodes);
  build_grid("nesw", rows, cols, nesw_grid);
  skel->orientation(2, nesw_grid);

  DPAG* swne_grid = new DPAG(_num_nodes);
  build_grid("swne", rows, cols, swne_grid);
  skel->orientation(3, swne_grid);

  _instance->skeleton(skel);
  
  return is;
}
