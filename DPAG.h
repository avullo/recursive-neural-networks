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
unsigned int max_indegree(const DPAG&);
unsigned int max_outdegree(const DPAG&);

// Function to produce a topological ordering of the nodes of a DPAG
std::vector<int> topological_sort(const DPAG&);

// Function to construct the grids corresponding
// to the four processing direction of a Recursive Neural Network
// applid to bidimensional grid domains
void build_grid(const std::string&, int, int, DPAG*);

// print DPAG to output stream
void printDPAG(const DPAG& dpag, std::ostream&);

#endif // _DPAG_H_
