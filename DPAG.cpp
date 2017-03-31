#include "General.h"
#include "DPAG.h"
#include <cassert>
#include <boost/graph/topological_sort.hpp>
using namespace std;

bool equal(const DPAG& g1, const DPAG& g2) {
  if(boost::num_vertices(g1) != boost::num_vertices(g2))
    return false;

  VertexId vertex_id = boost::get(boost::vertex_index, g1);
  vertexIt v_i, v_end;
  outIter out_i, out_i1, out_end, out_end1;
  adjIter adj_i, adj_i1, adj_end, adj_end1;
  cEdgeId edge_id1 = boost::get(boost::edge_index, g1);
  cEdgeId edge_id2 = boost::get(boost::edge_index, g2);
  
  for(boost::tie(v_i, v_end) = boost::vertices(g1); v_i!=v_end; ++v_i) {
    int num_nodes_g1 = boost::out_degree(*v_i, g1), num_nodes_g2 = boost::out_degree(boost::vertex(vertex_id[*v_i], g2), g2);
    if(num_nodes_g1 != num_nodes_g2)
      return false;
    else {
      // int index = 0;
      // vector<int> children_1(num_nodes_g1);//, children_2;
      // for(boost::tie(out_i, out_end)=boost::out_edges(*v_i, g1);
      // 	  out_i!=out_end; ++out_i) {
      // 	children_1[index++] = boost::target(*out_i, g1);
      // 	//children_1.push_back(boost::target(*out_i, g1));
      // }
      // for(boost::tie(out_i, out_end)=boost::out_edges(boost::vertex(vertex_id[*v_i], g2), g2);
      // 	  out_i!=out_end; ++out_i) {
      // 	//if(find(children_1.begin(), children_1.end(), boost::target(*out_i, g2)) == children_1.end())
      // 	int target_g2 = boost::target(*out_i, g2);
      // 	bool found = false;
      // 	for(int i=0; i<children_1.size(); i++) {
      // 	  if(target_g2 == children_1[i]) {
      // 	    found = true;
      // 	    break;
      // 	  }
      // 	}
      // 	if(!found)
      // 	  return false;
      // }
      for(boost::tie(out_i, out_end)=boost::out_edges(*v_i, g1); out_i!=out_end; ++out_i) {
	if(not boost::edge(boost::source(*out_i, g1), boost::target(*out_i, g1), g2).second)
	  return false;
	Edge_d e = boost::edge(boost::source(*out_i, g1), boost::target(*out_i, g1), g2).first;
	if(edge_id2[e] !=
	   edge_id1[*out_i]) return false;
      }
    }
  }
  return true;
}

unsigned int max_indegree(const DPAG& dpag) {
  vertexIt v_i, v_end;
  
  int max_indegree = 0;
  for(boost::tie(v_i, v_end) = boost::vertices(dpag); v_i!=v_end; ++v_i) {
    int current_node_indegree = boost::in_degree(*v_i, dpag);
    if(max_indegree < current_node_indegree)
      max_indegree = current_node_indegree;
  }

  return max_indegree;
}

unsigned int max_outdegree(const DPAG& dpag) {
  //VertexId vertex_id = boost::get(boost::vertex_index, dpag);
  vertexIt v_i, v_end;
  
  int max_outdegree = 0;
  for(boost::tie(v_i, v_end) = boost::vertices(dpag); v_i!=v_end; ++v_i) {
    int current_node_outdegree = boost::out_degree(*v_i, dpag);
    if(max_outdegree < current_node_outdegree)
      max_outdegree = current_node_outdegree;
  }

  return max_outdegree;
}

std::vector<int> topological_sort(const DPAG& dpag) {
  std::vector<Vertex_d> reverse_linear_ordering;
  boost::topological_sort(dpag, std::back_inserter(reverse_linear_ordering));

  // The previous call produce a reverse topological ordering,
  // so we must perform the following...
  // std::vector<Vertex_d> linear_ordering(reverse_linear_ordering.rbegin(),
  // 					reverse_linear_ordering.rend());

  vector<int> linear_ordering;
  VertexId vertex_id = boost::get(boost::vertex_index, dpag);
  // TODO: use iterators
  for(int i=reverse_linear_ordering.size()-1; i>=0; --i) {
    linear_ordering.push_back(vertex_id[reverse_linear_ordering[i]]);
  }
  return linear_ordering;
}

void build_grid(const std::string& direction, int rows, int cols, DPAG& dpag, std::vector<int>& top_ord) {
  int num_nodes = rows * cols;
  assert(boost::num_vertices(dpag) == (uint)num_nodes &&  
	 boost::num_edges(dpag) == 0 &&
	 top_ord.size() == (uint)num_nodes);

  if(direction == "nwse") {
    for(int i=0; i<rows; ++i) {
      for(int j=0; j<cols; ++j) {
	int edge_index = 0;
	if(j<cols-1)
	  boost::add_edge(i*rows+j, i*rows+j+1, EdgeProperty(edge_index++), dpag);
	if(i<rows-1)
	  boost::add_edge(i*rows+j, (i+1)*rows+j, EdgeProperty(edge_index++), dpag);
      }
    }
    
    int index = 0;
    for(int j=0; j<cols; ++j) {
      for(int i=0; i<rows; ++i) {
	top_ord[index++] = i*rows+j;
      }
    }
  } else if(direction == "senw") {
    for(int i=rows-1; i>=0; --i) {
      for(int j=cols-1; j>=0; --j) {
	int edge_index = 0;
	if(j>0)
	  boost::add_edge(i*rows+j, i*rows+j-1, EdgeProperty(edge_index++), dpag);
	if(i>0)
	  boost::add_edge(i*rows+j, (i-1)*rows+j, EdgeProperty(edge_index++), dpag);
      }
    }
    
    int index = 0;
    for(int i=rows-1; i>=0; --i) {
      for(int j=cols-1; j>=0; --j) {
	top_ord[index++] = i*rows+j;
      }
    }
  } else if(direction == "nesw") {
    for(int i=0; i<rows; ++i) {
      for(int j=cols-1; j>=0; --j) {
	int edge_index = 0;
	if(j>0)
	  boost::add_edge(i*rows+j, i*rows+j-1, EdgeProperty(edge_index++), dpag);
	if(i<rows-1)
	  boost::add_edge(i*rows+j, (i+1)*rows+j, EdgeProperty(edge_index++), dpag);
      }
    }
    
    int index = 0;
    for(int j=cols-1; j>=0; --j) {
      for(int i=0; i<rows; ++i) {
	top_ord[index++] = i*rows+j;
      }
    }
  } else if(direction == "swne") {
    for(int i=rows-1; i>=0; --i) {
      for(int j=0; j<cols; ++j) {
	int edge_index = 0;
	if(j<cols-1)
	  boost::add_edge(i*rows+j, i*rows+j+1, EdgeProperty(edge_index++), dpag);
	if(i>0)
	  boost::add_edge(i*rows+j, (i-1)*rows+j, EdgeProperty(edge_index++), dpag);
      }
    }
    
    int index = 0;
    for(int i=rows-1; i>=0; --i) {
      for(int j=0; j<cols; ++j) {
	top_ord[index++] = i*rows+j;
      }
    }

  } else {
    cerr << "ERROR! Wrong direction...\n";
    exit(1);
  }
}

void print(const DPAG& dpag, ostream& out) {
  VertexId vertex_id = boost::get(boost::vertex_index, dpag);
  //EdgeId edge_id = boost::get(edge_ordered_tuple_index_t(), dpag);
  cEdgeId edge_id = boost::get(boost::edge_index, dpag);
  vertexIt v_i, v_i1, v_end;
  outIter out_i, out_end;
  for(boost::tie(v_i, v_end) = boost::vertices(dpag); v_i!=v_end; ++v_i) {
    out << vertex_id[*v_i] << " --> ";
    for(boost::tie(out_i, out_end)=boost::out_edges(*v_i, dpag);
	out_i!=out_end; ++out_i) {
      out << vertex_id[boost::target(*out_i, dpag)] << " ("
	  << edge_id[*out_i] << "), ";
    }
    out << endl;
  }
}
