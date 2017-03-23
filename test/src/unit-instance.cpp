#include "catch.hpp"

#include "Instance.h"
#include "InstanceParser.h"
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic tests", "[instance]") {
  // prepare arguments and read configuration file
  setenv("RNNOPTIONTYPE", "train", 1);
  char* argv[] = { (char*)"dummy", (char*)"-c", (char*)"data/rnn.conf" };
  Options::instance()->parse_args(3, argv);

  vertexIt v_i, v_end;
  outIter out_i, out_end;
  ieIter in_i, in_end;

  // build and test sequence
  SECTION("sequence") {
    InstanceParser p;
    ifstream is("data/sequence.gph");
    Instance* instance = p.read(is);
    is.close();

    CHECK(instance->id() == "dummy");
    CHECK(instance->domain() == SEQUENCE);
    CHECK(instance->transduction() == IO_ISOMORPH);
    CHECK_FALSE(instance->output_dim());

    SECTION("accessing nodes") {
      vector<float> node_input, node_target;
      Node* n = instance->node(0);
      node_input = n->input();
      node_target = n->target();
      CHECK(node_input[0] == .1f); CHECK(node_input[1] == .2f); CHECK(node_input[2] == .3f);
      CHECK(node_target[0] == .3f); CHECK(node_target[1] == .2f); CHECK(node_target[2] == .1f);

      n = instance->node(4);
      node_input = n->input();
      node_target = n->target();
      CHECK(node_input[0] == .3f); CHECK(node_input[1] == .4f); CHECK(node_input[2] == .5f);
      CHECK(node_target[0] == .5f); CHECK(node_target[1] == .4f); CHECK(node_target[2] == .3f);
    }
    
    SECTION("accessing skeleton") {
      CHECK(instance->maximum_indegree() == 1);
      CHECK(instance->maximum_outdegree() == 1);
      CHECK(instance->num_orient() == 1);

      // CHECK_THROWS(instance->orientation(1));
      DPAG* sequence = instance->orientation(0);

      int T = boost::num_vertices(*sequence);
      VertexId vertex_id = boost::get(boost::vertex_index, *sequence);
      EdgeId edge_id = boost::get(boost::edge_index, *sequence);
    
      for(boost::tie(v_i, v_end) = boost::vertices(*sequence); v_i!=v_end; ++v_i) {
	boost::tie(out_i, out_end)=boost::out_edges(*v_i, *sequence);
	boost::tie(in_i, in_end)=boost::in_edges(*v_i, *sequence);
      
	int id = vertex_id[*v_i];
	if(id == 0) {
	  CHECK(boost::source(*in_i, *sequence) == vertex_id[*(v_i+1)]);
	  CHECK(++in_i == in_end);
	  CHECK(out_i == out_end);
	} else if(id < T-1) {
	  CHECK(boost::target(*out_i, *sequence) == vertex_id[*(v_i-1)]);
	  CHECK(++out_i == out_end);
	  CHECK(boost::source(*in_i, *sequence) == vertex_id[*(v_i+1)]);
	  CHECK(++in_i == in_end);
	} else {
	  CHECK(id == T-1);
	  CHECK(boost::target(*out_i, *sequence) == vertex_id[*(v_i-1)]);
	  CHECK(++out_i == out_end);
	  CHECK(in_i == in_end);
	}
      }

      SECTION("topological sort") {
	// CHECK_THROWS(instance->topological_order(1));
	vector<int> top_sort = instance->topological_order(0);

	CHECK(top_sort.size() == boost::num_vertices(*sequence));
	for(uint i=0; i<boost::num_vertices(*sequence); ++i)
	  CHECK(top_sort[i] == T-i-1);
      }
    }
  }

  // HERE: build and test linear chain
  SECTION("linear chain") {
    Options::instance()->domain(LINEARCHAIN);
    InstanceParser p;
    ifstream is("data/sequence.gph");
    Instance* instance = p.read(is);
    is.close();

    CHECK(instance->id() == "dummy");
    CHECK(instance->domain() == LINEARCHAIN);
    CHECK(instance->transduction() == IO_ISOMORPH);
    CHECK_FALSE(instance->output_dim());

    SECTION("accessing nodes") {
      vector<float> node_input, node_target;
      Node* n = instance->node(0);
      node_input = n->input();
      node_target = n->target();
      CHECK(node_input[0] == .1f); CHECK(node_input[1] == .2f); CHECK(node_input[2] == .3f);
      CHECK(node_target[0] == .3f); CHECK(node_target[1] == .2f); CHECK(node_target[2] == .1f);

      n = instance->node(4);
      node_input = n->input();
      node_target = n->target();
      CHECK(node_input[0] == .3f); CHECK(node_input[1] == .4f); CHECK(node_input[2] == .5f);
      CHECK(node_target[0] == .5f); CHECK(node_target[1] == .4f); CHECK(node_target[2] == .3f);
    }
    
    SECTION("accessing skeleton") {
      CHECK(instance->maximum_indegree() == 1);
      CHECK(instance->maximum_outdegree() == 1);
      CHECK(instance->num_orient() == 2);

      // CHECK_THROWS(instance->orientation(2));
      DPAG* lrseq = instance->orientation(1);
      int T = boost::num_vertices(*lrseq);
      VertexId vertex_id = boost::get(boost::vertex_index, *lrseq);
      EdgeId edge_id = boost::get(boost::edge_index, *lrseq);
    
      for(boost::tie(v_i, v_end) = boost::vertices(*lrseq); v_i!=v_end; ++v_i) {
  	boost::tie(out_i, out_end)=boost::out_edges(*v_i, *lrseq);
  	boost::tie(in_i, in_end)=boost::in_edges(*v_i, *lrseq);
      
  	int id = vertex_id[*v_i];
  	if(id == 0) {
  	  CHECK(boost::target(*out_i, *lrseq) == vertex_id[*(v_i+1)]);
  	  CHECK(++out_i == out_end);
  	  CHECK(in_i == in_end);
  	} else if(id < T-1) {
  	  CHECK(boost::target(*out_i, *lrseq) == vertex_id[*(v_i+1)]);
  	  CHECK(++out_i == out_end);
  	  CHECK(boost::source(*in_i, *lrseq) == vertex_id[*(v_i-1)]);
  	  CHECK(++in_i == in_end);
  	} else {
  	  CHECK(id == T-1);
  	  CHECK(boost::source(*in_i, *lrseq) == vertex_id[*(v_i-1)]);
  	  CHECK(++in_i == in_end);
  	  CHECK(out_i == out_end);
  	}
      }

      SECTION("topological sort") {
  	// CHECK_THROWS(instance->topological_order(2));
  	vector<int> top_sort = instance->topological_order(1);

  	CHECK(top_sort.size() == boost::num_vertices(*lrseq));
  	for(uint i=0; i<boost::num_vertices(*lrseq); ++i)
  	  CHECK(top_sort[i] == i);
      }
    }
  }


}
