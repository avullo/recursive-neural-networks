#ifndef _DPAG_H_
#define _DPAG_H_

/*

  Declaration of a Direct Positional Acyclic Graph (DPAG)

  A DPAG is the fundamental data structure which is processed
  by one or more of the various RNN state transition networks.

 */

#include <utility> // for std::pair
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp> 
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <iostream>

typedef boost::property<boost::edge_index_t, unsigned int> EdgeProperty;
typedef boost::property<boost::vertex_index_t, unsigned int> VertexProperty;

// Create a typedef for the DPAG type. We can change it according to space and time complexity needs.
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, VertexProperty, EdgeProperty> DPAG;

// Other related useful type declarations...
typedef boost::graph_traits<DPAG>::vertex_descriptor Vertex_d;
typedef boost::graph_traits<DPAG>::vertices_size_type Vst;
typedef boost::graph_traits<DPAG>::vertex_iterator vertexIt;
typedef boost::property_map<DPAG, boost::vertex_index_t>::type VertexId;
typedef boost::property_map<DPAG, boost::edge_index_t>::type EdgeId;
typedef boost::property_map<DPAG, boost::edge_index_t>::const_type cEdgeId;

typedef std::pair<int, int> Edge;
typedef boost::graph_traits<DPAG>::edge_descriptor Edge_d;
typedef boost::graph_traits<DPAG>::edge_iterator eIter;
typedef boost::graph_traits<DPAG>::in_edge_iterator ieIter;
typedef boost::graph_traits<DPAG>::out_edge_iterator outIter;
typedef boost::graph_traits<DPAG>::adjacency_iterator adjIter;

// Check if two DPAGs are equal
bool equal(const DPAG&, const DPAG&);

// Functions to return maximum indegree/outdegree of a given graph
unsigned int get_max_indegree(const DPAG&);
unsigned int get_max_outdegree(const DPAG&);

// Function to produce a topological ordering of the nodes of a DPAG
std::vector<Vertex_d> topological_sort(const DPAG&);
  
// print DPAG to output stream
void print(const DPAG& dpag, std::ostream&);

#endif // _DPAG_H_
