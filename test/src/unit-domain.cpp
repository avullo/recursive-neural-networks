#include "catch.hpp"

#include "DPAG.h"
#include "StructuredDomain.h"
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic facs about a domain", "[structured domain]") {
  Domain domains[] = { DOAG, SEQUENCE, LINEARCHAIN, NARYTREE, UG, GRID2D };
  int orientations[] = { 1, 1, 2, 1, 2, 4 };

  for(int i=0; i<6; ++i)
    CHECK(num_orientations(domains[i]) == orientations[i]);
  CHECK_THROW(num_orientations(0));
  CHECK_THORW(num_orientations(6));

  
	  

  // uint num_nodes = 10;
  // DPAG d(num_nodes);
  
  // CHECK(boost::num_vertices(d) == num_nodes);
  
  // Vertex_d v;
  // VertexId vertex_id = boost::get(boost::vertex_index, d);
  // vertexIt v_i, v_end;
  // outIter out_i, out_end, out_i1;
  // EdgeId edge_id = boost::get(boost::edge_index, d);

  // // access vertices with iterators and indices
  // SECTION("accessing vertices") {
  //   boost::tie(v_i, v_end) = boost::vertices(d);
  //   SECTION("first vertex") {
  //     CHECK(vertex_id[*v_i] == 0);
  //   }
  //   SECTION("another vertex") {
  //     v_i++; v_i++; v_i++;
  //     CHECK(vertex_id[*v_i] == 3);
  //   }
  // }
  
  // // check looping over edges
  // // add edges, check results with iterators
  // SECTION("add some edges") {
  //   boost::add_edge(2,3, EdgeProperty(0), d);
  //   boost::add_edge(2,8, EdgeProperty(1), d);
  //   CHECK(boost::num_edges(d) == 2);

  //   // we've added the right edge
  //   v = boost::vertex(2, d);
  //   boost::tie(out_i, out_end)=boost::out_edges(v, d);
  //   SECTION("accessing first edge") {
  //     CHECK(target(*out_i, d) == 3);
  //     CHECK(edge_id[*out_i] == 0);
  //   }

  //   SECTION("accessing second and last edge") {
  //     CHECK(target(*(++out_i), d) == 8);
  //     CHECK(edge_id[*out_i] == 1);
  //     CHECK(++out_i == out_end);
  //   }
  // }
}

// class DPAGGenerator {
//   DPAG** dpags;
  
// public:

//   // build a sequence of T steps
//   DPAG make_sequence(uint T) {
//     DPAG s(T);

//     // make the connections go from right to left
//     // to emulate RNN unfolding from right to left,
//     // i.e. reverse topological sort
//     for(uint i=T-1; i>0; --i)
//       boost::add_edge(i, i-1, s);
    
//     return s;
//   }
  
//   DPAG make_dpag(const char* fname) {
//     ifstream is(fname);
//     if(!is) {
//       fprintf(stderr, "Could not open file %s\n", fname);
//       exit(1);
//     }

//     string line;
//     getline(is, line);
//     istringstream iss(line);
//     int V;
//     iss >> V;
//     DPAG d(V);
    
//     for(int i=0; i<V; ++i) {
//       getline(is, line);
//       istringstream iss1(line);
//       int v;
//       iss1 >> v;
//       int target;
//       int eindex = 0;
//       while(iss1 >> target)
// 	boost::add_edge(v, target, EdgeProperty(eindex++), d);
//     }

//     return d;
//   }

  
//   DPAG make_grid(uint N) {
//     DPAG grid(N);

//     // build a nwse grid
//     for(uint i=0; i<N; ++i) {
//       for(uint j=0; j<N; ++j) {
// 	int edge_index = 0;
// 	if(j<N-1)
// 	  boost::add_edge(i*N+j, i*N+j+1, EdgeProperty(edge_index++), grid);
// 	if(i<N-1)
// 	  boost::add_edge(i*N+j, (i+1)*N+j, EdgeProperty(edge_index++), grid);
//       }
//     }

//     return grid;
//   }
// };
  
// TEST_CASE("DPAG functions with various data structures", "[dpag]") {
//   //Vertex_d v;
//   vertexIt v_i, v_end;
//   outIter out_i, out_end;
//   ieIter in_i, in_end;

//   DPAGGenerator dg;

//   int T = 10;
//   DPAG sequence = dg.make_sequence(T);
  
//   CHECK(boost::num_vertices(sequence) == T);

//   DPAG dpag = dg.make_dpag("data/test_dpag.gph");
  
//   int N = T;
//   DPAG grid = dg.make_grid(N);
//   CHECK(boost::num_vertices(grid) == N*N);

//   SECTION("equal - same skeleton") {
//     DPAG sequence1 = dg.make_sequence(T);
//     DPAG dpag1 = dg.make_dpag("data/test_dpag.gph");
//     DPAG grid1 = dg.make_grid(N);
    
//     SECTION("same edge properties") {
//       CHECK(equal(sequence, sequence1));  
//       CHECK(equal(dpag, dpag1));
//       CHECK(equal(grid, grid1));
      
//       SECTION("different edge properties") {
// 	DPAG sequence2 = dg.make_sequence(T);
	
// 	VertexId vertex_id = boost::get(boost::vertex_index, sequence2);
// 	Vertex_d v = boost::vertex(3, sequence2);
// 	outIter out_i = boost::out_edges(v, sequence2).first;
// 	EdgeId edge_id = boost::get(boost::edge_index, sequence2);
// 	edge_id[*out_i] = 2;
// 	CHECK_FALSE(equal(sequence, sequence2));
//       }
//     }
//   }

//   SECTION("equal - different skeleton") {
//     CHECK_FALSE(equal(sequence, dpag));
//     CHECK_FALSE(equal(sequence, grid));
//     CHECK_FALSE(equal(dpag, grid));
//   }

//   SECTION("linear sequence") {
//     CHECK(max_indegree(sequence) == 1);
//     CHECK(max_outdegree(sequence) == 1);
    
//     VertexId vertex_id = boost::get(boost::vertex_index, sequence);
//     EdgeId edge_id = boost::get(boost::edge_index, sequence);
    
//     for(boost::tie(v_i, v_end) = boost::vertices(sequence); v_i!=v_end; ++v_i) {
//       boost::tie(out_i, out_end)=boost::out_edges(*v_i, sequence);
//       boost::tie(in_i, in_end)=boost::in_edges(*v_i, sequence);
      
//       int id = vertex_id[*v_i];
//       if(id == 0) {
// 	CHECK(boost::source(*in_i, sequence) == vertex_id[*(v_i+1)]);
// 	CHECK(++in_i == in_end);
// 	CHECK(out_i == out_end);
//       } else if(id < T-1) {
// 	CHECK(boost::target(*out_i, sequence) == vertex_id[*(v_i-1)]);
// 	CHECK(++out_i == out_end);
// 	CHECK(boost::source(*in_i, sequence) == vertex_id[*(v_i+1)]);
// 	CHECK(++in_i == in_end);
//       } else {
// 	CHECK(id == T-1);
// 	CHECK(boost::target(*out_i, sequence) == vertex_id[*(v_i-1)]);
// 	CHECK(++out_i == out_end);
// 	CHECK(in_i == in_end);
//       }
//     }

//     SECTION("topological sort") {
//       std::vector<int> top_sort = topological_sort(sequence);
//       CHECK(top_sort.size() == boost::num_vertices(sequence));
//       for(uint i=0; i<boost::num_vertices(sequence); ++i)
// 	CHECK(top_sort[i] == T-i-1);
//     }
//   }

//   SECTION("DPAG") {
//     CHECK(max_indegree(dpag) == 2);
//     CHECK(max_outdegree(dpag) == 2);

//     VertexId vertex_id = boost::get(boost::vertex_index, dpag);
//     EdgeId edge_id = boost::get(boost::edge_index, dpag);

//     Vertex_d v = boost::vertex(0, dpag);
//     boost::tie(out_i, out_end)=boost::out_edges(v, dpag);
//     boost::tie(in_i, in_end)=boost::in_edges(v, dpag);

//     CHECK(in_i == in_end);
//     CHECK(boost::target(*out_i, dpag) == 1);
//     CHECK(edge_id[*out_i] == 0);
//     CHECK(boost::target(*(++out_i), dpag) == 3);
//     CHECK(edge_id[*out_i] == 1);
//     CHECK(++out_i == out_end);

//     v = boost::vertex(1, dpag);
//     boost::tie(out_i, out_end)=boost::out_edges(v, dpag);
//     boost::tie(in_i, in_end)=boost::in_edges(v, dpag);
//     CHECK(boost::source(*in_i, dpag) == 0);
//     CHECK(++in_i == in_end);
//     CHECK(boost::target(*out_i, dpag) == 2);
//     CHECK(edge_id[*out_i] == 0);
//     CHECK(boost::target(*(++out_i), dpag) == 3);
//     CHECK(edge_id[*out_i] == 1);
//     CHECK(++out_i == out_end);

//     v = boost::vertex(2, dpag);
//     boost::tie(out_i, out_end)=boost::out_edges(v, dpag);
//     boost::tie(in_i, in_end)=boost::in_edges(v, dpag);
//     CHECK(boost::source(*in_i, dpag) == 1);
//     CHECK(++in_i == in_end);
//     CHECK(out_i == out_end);

//     v = boost::vertex(3, dpag);
//     boost::tie(out_i, out_end)=boost::out_edges(v, dpag);
//     boost::tie(in_i, in_end)=boost::in_edges(v, dpag);
//     CHECK(boost::source(*in_i, dpag) == 0);
//     CHECK(boost::source(*(++in_i), dpag) == 1);
//     CHECK(++in_i == in_end);
//     CHECK(out_i == out_end);

//     SECTION("topological sort") {
//       vector<int> top_sort = topological_sort(dpag);
//       int V = boost::num_vertices(dpag);
//       CHECK(top_sort.size() == V);
      
//       CHECK(top_sort[0] == 0);
//       CHECK(top_sort[1] == 1);
      
//       // NOTE: cannot check with complex expressions
//       // Is there a better way?
//       CHECK(top_sort[2] >= 2);
//       CHECK(top_sort[2] <= 3);
//       CHECK(top_sort[3] >= 2);
//       CHECK(top_sort[3] <= 3);
//     }
//   }

//   SECTION("Grid") {
//     CHECK(max_indegree(grid) == 2);
//     CHECK(max_outdegree(grid) == 2);

//     VertexId vertex_id = boost::get(boost::vertex_index, grid);
//     EdgeId edge_id = boost::get(boost::edge_index, grid);

//     Vertex_d v = boost::vertex(0, grid);
//     boost::tie(out_i, out_end)=boost::out_edges(v, grid);
//     boost::tie(in_i, in_end)=boost::in_edges(v, grid);

//     v = boost::vertex(0, grid);
//     boost::tie(out_i, out_end)=boost::out_edges(v, grid);
//     boost::tie(in_i, in_end)=boost::in_edges(v, grid);
//     CHECK(in_i == in_end);
//     CHECK(boost::target(*out_i, grid) == 1);
//     CHECK(edge_id[*out_i] == 0);
//     CHECK(boost::target(*(++out_i), grid) == 10);
//     CHECK(edge_id[*out_i] == 1);
//     CHECK(++out_i == out_end);

//     v = boost::vertex(9, grid);
//     boost::tie(out_i, out_end)=boost::out_edges(v, grid);
//     boost::tie(in_i, in_end)=boost::in_edges(v, grid);
//     CHECK(boost::source(*in_i, grid) == 8);
//     CHECK(++in_i == in_end);
//     CHECK(boost::target(*out_i, grid) == 19);
//     CHECK(edge_id[*out_i] == 0);
//     CHECK(++out_i == out_end);

//     v = boost::vertex(14, grid);
//     boost::tie(out_i, out_end)=boost::out_edges(v, grid);
//     boost::tie(in_i, in_end)=boost::in_edges(v, grid);
//     CHECK(boost::source(*in_i, grid) == 4);
//     CHECK(boost::source(*(++in_i), grid) == 13);
//     CHECK(++in_i == in_end);
//     CHECK(boost::target(*out_i, grid) == 15);
//     CHECK(edge_id[*out_i] == 0);
//     CHECK(boost::target(*(++out_i), grid) == 24);
//     CHECK(edge_id[*out_i] == 1);
//     CHECK(++out_i == out_end);
    
//     SECTION("topological sort") {
//       vector<int> top_sort = topological_sort(grid);

//       CHECK(top_sort.size() == N*N);
//       CHECK(top_sort[0] == 0);
//       CHECK(top_sort[N*N-1] == N*N-1);

//       // check other relative orders
//       // copy(top_sort.begin(), top_sort.end(), ostream_iterator<int>(cout, " "));

//       // information flows NWSE
//       vector<int>::iterator it;
//       vector<int>::iterator it1;
//       vector<int>::iterator it2;
//       for(int i=0; i<N-1; ++i) {
// 	const vector<int>::size_type j = find(top_sort.begin(), top_sort.end(), i*N+i) - top_sort.begin();
// 	const vector<int>::size_type j1 = find(top_sort.begin(), top_sort.end(), i*N+i+1) - top_sort.begin();
// 	const vector<int>::size_type j2 = find(top_sort.begin(), top_sort.end(), (i+1)*N+i) - top_sort.begin();

// 	CHECK(j < j1);
// 	CHECK(j < j2);
// 	CHECK(j2 < j1);
//       }
      
//     }
//   }
// }
