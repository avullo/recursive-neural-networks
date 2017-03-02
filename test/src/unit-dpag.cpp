#include "catch.hpp"

#include "DPAG.h"
#include <vector>
#include <iostream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic DPAG usage", "[dpag]") {
  uint num_nodes = 10;
  DPAG d(num_nodes);
  
  CHECK(boost::num_vertices(d) == num_nodes);
  
  Vertex_d v;
  VertexId vertex_id = boost::get(boost::vertex_index, d);
  vertexIt v_i, v_end;
  outIter out_i, out_end, out_i1;
  EdgeId edge_id = boost::get(boost::edge_index, d);

  SECTION("accessing vertices") {
    boost::tie(v_i, v_end) = boost::vertices(d);
    SECTION("first vertex") {
      CHECK(vertex_id[*v_i] == 0);
    }
    SECTION("another vertex") {
      v_i++; v_i++; v_i++;
      CHECK(vertex_id[*v_i] == 3);
    }
  }
  
  // check looping over edges
  // add edges, check results with iterators
  SECTION("add some edges") {
    boost::add_edge(2,3, EdgeProperty(0), d);
    boost::add_edge(2,8, EdgeProperty(1), d);
    CHECK(boost::num_edges(d) == 2);

    // we've added the right edge
    v = boost::vertex(2, d);
    boost::tie(out_i, out_end)=boost::out_edges(v, d);
    SECTION("accessing first edge") {
      CHECK(target(*out_i, d) == 3);
      CHECK(edge_id[*out_i] == 0);
    }

    SECTION("accessing second and last edge") {
      CHECK(target(*(++out_i), d) == 8);
      CHECK(edge_id[*out_i] == 1);
      CHECK(++out_i == out_end);
    }
  }
}

// class DPAGGenerator {
//   DPAG* dpags;
  
// public:
//   DPAGGenerator() {
//     make_sequence("../data/sequence.gph");
//     make_linear_chain("../data/sequence.gph");
//   }
  
//   void make_sequence(const char* fname) {
//   }
    
//   void make_linear_chain(const char* fname) {
//   }
// };
  
// TEST_CASE("Different data structures", "[dpag]") {
// }
