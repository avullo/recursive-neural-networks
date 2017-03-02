#include "DPAG.h"
#include <boost/graph/topological_sort.hpp>
using namespace std;

bool equal(const DPAG& g1, const DPAG& g2) {
  if(boost::num_vertices(g1) != boost::num_vertices(g2))
    return false;

  Vertex_d node;
  VertexId vertex_id = boost::get(boost::vertex_index, g1);
  vertexIt v_i, v_end;
  outIter out_i, out_i1, out_end, out_end1;
  adjIter adj_i, adj_i1, adj_end, adj_end1;

  for(boost::tie(v_i, v_end) = boost::vertices(g1); v_i!=v_end; ++v_i) {
    int num_nodes_g1 = boost::out_degree(*v_i, g1), num_nodes_g2 = boost::out_degree(boost::vertex(vertex_id[*v_i], g2), g2);
    if(num_nodes_g1 != num_nodes_g2)
      return false;
    else {
      int index = 0;
      vector<int> children_1(num_nodes_g1);//, children_2;
      for(boost::tie(out_i, out_end)=boost::out_edges(*v_i, g1);
	  out_i!=out_end; ++out_i) {
	children_1[index++] = boost::target(*out_i, g1);
	//children_1.push_back(boost::target(*out_i, g1));
      }
      for(boost::tie(out_i, out_end)=boost::out_edges(boost::vertex(vertex_id[*v_i], g2), g2);
	  out_i!=out_end; ++out_i) {
	//if(find(children_1.begin(), children_1.end(), boost::target(*out_i, g2)) == children_1.end())
	int target_g2 = boost::target(*out_i, g2);
	bool found = false;
	for(int i=0; i<children_1.size(); i++) {
	  if(target_g2 == children_1[i]) {
	    found = true;
	    break;
	  }
	}
	if(!found)
	  return false;
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

std::vector<Vertex_d> topological_sort(DPAG& dpag) {
  std::vector<Vertex_d> reverse_linear_ordering;
  boost::topological_sort(dpag, std::back_inserter(reverse_linear_ordering));

  // The previous call produce a reverse topological ordering,
  // so we must perform the following...
  std::vector<Vertex_d> linear_ordering(reverse_linear_ordering.rbegin(),
					reverse_linear_ordering.rend());
  return linear_ordering;
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
