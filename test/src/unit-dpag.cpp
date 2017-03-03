#include "catch.hpp"

#include "DPAG.h"
#include <cstdio>
#include <vector>
#include <fstream>
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

  // access vertices with iterators and indices
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

class DPAGGenerator {
  DPAG** dpags;
  
public:

  // build a sequence of T steps
  DPAG make_sequence(uint T) {
    DPAG s(T);

    // make the connections go from right to left
    // to emulate RNN unfolding from right to left,
    // i.e. reverse topological sort
    for(uint i=T-1; i>0; --i)
      boost::add_edge(i, i-1, s);
    
    return s;
  }
  
  DPAG make_dpag(const char* fname) {
    // ifstream is(fname);
    // if(!is) {
    //   fprintf(stderr, "Could not open file %s\n", fname);
    //   exit(1);
    // }

    DPAG d;

    return d;
  }

  
  DPAG make_grid(uint N) {
    DPAG grid(N);

    return grid;
  }
};
  
TEST_CASE("DPAG functions with various data structures", "[dpag]") {
  //Vertex_d v;
  vertexIt v_i, v_end;
  outIter out_i, out_end;
  ieIter in_i, in_end;

  DPAGGenerator dg;

  int T = 10;
  DPAG sequence = dg.make_sequence(T);
  
  CHECK(boost::num_vertices(sequence) == T);

  DPAG dpag = dg.make_dpag("../data/test_dpag.gph");
  
  int N = T;
  DPAG grid = dg.make_grid(N);
  CHECK(boost::num_vertices(grid) == N);

  SECTION("equal - same skeleton") {
    DPAG sequence1 = dg.make_sequence(T);
    CHECK(equal(sequence, sequence1));

    DPAG dpag1 = dg.make_dpag("../data/test_dpag.gph");
    CHECK(equal(dpag, dpag1));

    DPAG grid1 = dg.make_grid(N);
    CHECK(equal(grid, grid1));

    SECTION("different edge properties") {
      DPAG sequence2 = dg.make_sequence(T);

      VertexId vertex_id = boost::get(boost::vertex_index, sequence2);
      Vertex_d v = boost::vertex(3, sequence2);
      outIter out_i = boost::out_edges(v, sequence2).first;
      EdgeId edge_id = boost::get(boost::edge_index, sequence2);
      edge_id[*out_i] = 2;
      CHECK_FALSE(equal(sequence, sequence2));
    }
  }

  SECTION("equal - different skeleton") {
    CHECK_FALSE(equal(sequence, dpag));
    CHECK_FALSE(equal(sequence, grid));
    CHECK_FALSE(equal(dpag, grid));
  }

  SECTION("sequence connectivity") {
    CHECK(max_indegree(sequence) == 1);
    CHECK(max_outdegree(sequence) == 1);
    
    VertexId vertex_id = boost::get(boost::vertex_index, sequence);
    EdgeId edge_id = boost::get(boost::edge_index, sequence);
    
    for(boost::tie(v_i, v_end) = boost::vertices(sequence); v_i!=v_end; ++v_i) {
      boost::tie(out_i, out_end)=boost::out_edges(*v_i, sequence);
      boost::tie(in_i, in_end)=boost::in_edges(*v_i, sequence);
      
      int id = vertex_id[*v_i];
      if(id == 0) {
	CHECK(boost::source(*in_i, sequence) == vertex_id[*(v_i+1)]);
	CHECK(++in_i == in_end);
	CHECK(out_i == out_end);
      } else if(id < T-1) {
	CHECK(boost::target(*out_i, sequence) == vertex_id[*(v_i-1)]);
	CHECK(++out_i == out_end);
	CHECK(boost::source(*in_i, sequence) == vertex_id[*(v_i+1)]);
	CHECK(++in_i == in_end);
      } else {
	CHECK(id == T-1);
	CHECK(boost::target(*out_i, sequence) == vertex_id[*(v_i-1)]);
	CHECK(++out_i == out_end);
	CHECK(in_i == in_end);
      }
    }
  }

  SECTION("DPAG connectivity") {
  }

  SECTION("Grid connectivity") {
  }
}
