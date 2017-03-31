#include "catch.hpp"

#include "Instance.h"
#include "InstanceParser.h"
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic instance tests", "[instance]") {
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
      CHECK(T == instance->num_nodes());
      VertexId vertex_id = boost::get(boost::vertex_index, *sequence);
    
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

  // build and test linear chain
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
      CHECK(T == instance->num_nodes());
      VertexId vertex_id = boost::get(boost::vertex_index, *lrseq);
    
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

  // build and test DOAG
  SECTION("DOAG") {
    Options::instance()->domain(DOAG);
    Options::instance()->transduction(SUPER_SOURCE);
    InstanceParser p;
    ifstream is("data/dpag.gph");
    Instance* instance = p.read(is);
    is.close();

    CHECK(instance->id() == "dummy");
    CHECK(instance->domain() == DOAG);

    SECTION("accessing graph target") {
      CHECK(instance->transduction() == SUPER_SOURCE);
      CHECK(instance->output_dim() == 3);
      vector<float> target = instance->target();
      CHECK(target.size() == 3);
      CHECK(target[0] == .3f); CHECK(target[1] == .2f); CHECK(target[2] == .1f);
    }
    
    SECTION("accessing nodes") {
      vector<float> node_input;
      Node* n = instance->node(0);
      node_input = n->input();
      CHECK(node_input[0] == .1f); CHECK(node_input[1] == .2f); CHECK(node_input[2] == .3f);
      
      n = instance->node(2);
      node_input = n->input();
      CHECK(node_input[0] == .7f); CHECK(node_input[1] == .8f); CHECK(node_input[2] == .9f);
    }
    
    SECTION("accessing skeleton") {
      CHECK(instance->maximum_indegree() == 2);
      CHECK(instance->maximum_outdegree() == 2);
      CHECK(instance->num_orient() == 1);

      DPAG* doag = instance->orientation(0);
      int V = boost::num_vertices(*doag);
      CHECK(V == instance->num_nodes());
      
      EdgeId edge_id = boost::get(boost::edge_index, *doag);

      Vertex_d v = boost::vertex(0, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);

      CHECK(in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 1);
      CHECK(edge_id[*out_i] == 0);
      CHECK(boost::target(*(++out_i), *doag) == 3);
      CHECK(edge_id[*out_i] == 1);
      CHECK(++out_i == out_end);

      v = boost::vertex(1, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 0);
      CHECK(++in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 2);
      CHECK(edge_id[*out_i] == 0);
      CHECK(boost::target(*(++out_i), *doag) == 3);
      CHECK(edge_id[*out_i] == 1);
      CHECK(++out_i == out_end);

      v = boost::vertex(2, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 1);
      CHECK(++in_i == in_end);
      CHECK(out_i == out_end);

      v = boost::vertex(3, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 0);
      CHECK(boost::source(*(++in_i), *doag) == 1);
      CHECK(++in_i == in_end);
      CHECK(out_i == out_end);

    }

    SECTION("topological sort") {
      vector<int> top_sort = instance->topological_order(0);
      CHECK(top_sort.size() == instance->num_nodes());
      
      CHECK(top_sort[0] == 0);
      CHECK(top_sort[1] == 1);
      
      // NOTE: cannot check with complex expressions
      // Is there a better way?
      CHECK(top_sort[2] >= 2);
      CHECK(top_sort[2] <= 3);
      CHECK(top_sort[3] >= 2);
      CHECK(top_sort[3] <= 3);
    }
  }

  // build and test Undirected Graph
  SECTION("UG") {
    Options::instance()->domain(UG);
    Options::instance()->transduction(SUPER_SOURCE);
    InstanceParser p;
    ifstream is("data/dpag.gph");
    Instance* instance = p.read(is);
    is.close();

    CHECK(instance->id() == "dummy");
    CHECK(instance->domain() == UG);

    SECTION("accessing graph target") {
      CHECK(instance->transduction() == SUPER_SOURCE);
      CHECK(instance->output_dim() == 3);
      vector<float> target = instance->target();
      CHECK(target.size() == 3);
      CHECK(target[0] == .3f); CHECK(target[1] == .2f); CHECK(target[2] == .1f);
    }
    
    SECTION("accessing nodes") {
      vector<float> node_input;
      Node* n = instance->node(0);
      node_input = n->input();
      CHECK(node_input[0] == .1f); CHECK(node_input[1] == .2f); CHECK(node_input[2] == .3f);
      
      n = instance->node(2);
      node_input = n->input();
      CHECK(node_input[0] == .7f); CHECK(node_input[1] == .8f); CHECK(node_input[2] == .9f);
    }
    
    SECTION("accessing skeleton") {
      CHECK(instance->maximum_indegree() == 3);
      CHECK(instance->maximum_outdegree() == 3);
      CHECK(instance->num_orient() == 2);

      DPAG* doag = instance->orientation(1);
      int V = boost::num_vertices(*doag);
      CHECK(V == instance->num_nodes());
      
      EdgeId edge_id = boost::get(boost::edge_index, *doag);

      Vertex_d v = boost::vertex(0, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 1);
      CHECK(edge_id[*in_i] == 1);
      CHECK(boost::source(*(++in_i), *doag) == 3);
      CHECK(edge_id[*in_i] == 2);
      CHECK(++in_i == in_end);
      CHECK(out_i == out_end);

      v = boost::vertex(1, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 2);
      CHECK(edge_id[*in_i] == 1);
      CHECK(boost::source(*(++in_i), *doag) == 3);
      CHECK(edge_id[*in_i] == 1);
      CHECK(++in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 0);
      CHECK(edge_id[*out_i] == 1);
      CHECK(++out_i == out_end);

      v = boost::vertex(2, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 3);
      CHECK(edge_id[*in_i] == 0);
      CHECK(++in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 1);
      CHECK(edge_id[*out_i] == 1);
      CHECK(++out_i == out_end);

      v = boost::vertex(3, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 2);
      CHECK(edge_id[*out_i] == 0);
      CHECK(boost::target(*(++out_i), *doag) == 1);
      CHECK(edge_id[*out_i] == 1);
      CHECK(boost::target(*(++out_i), *doag) == 0);
      CHECK(edge_id[*out_i] == 2);
      CHECK(++out_i == out_end);
    }

    SECTION("topological sort") {
      vector<int> top_sort = instance->topological_order(1);
      CHECK(top_sort.size() == instance->num_nodes());
      
      CHECK(top_sort[0] == 3);
      CHECK(top_sort[1] == 2);
      
      // NOTE: cannot check with complex expressions
      // Is there a better way?
      CHECK(top_sort[2] >= 0);
      CHECK(top_sort[2] <= 1);
      CHECK(top_sort[3] >= 0);
      CHECK(top_sort[3] <= 1);
    }
  }

  SECTION("2DGRID") {
    Options::instance()->domain(GRID2D);
    Options::instance()->transduction(IO_ISOMORPH);
    InstanceParser p;
    ifstream is("data/grid.gph");
    Instance* instance = p.read(is);
    is.close();

    CHECK(instance->id() == "dummy");
    CHECK(instance->domain() == GRID2D);
    CHECK(instance->transduction() == IO_ISOMORPH);
    
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
      CHECK(instance->maximum_indegree() == 2);
      CHECK(instance->maximum_outdegree() == 2);
      CHECK(instance->num_orient() == 4);

      SECTION("NWSE") {
	DPAG* grid = instance->orientation(0);
	EdgeId edge_id = boost::get(boost::edge_index, *grid);

	Vertex_d v = boost::vertex(0, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 1);
	CHECK(edge_id[*out_i] == 0);
	CHECK(boost::target(*(++out_i), *grid) == 3);
	CHECK(edge_id[*out_i] == 1);
	CHECK(++out_i == out_end);

	v = boost::vertex(5, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::source(*in_i, *grid) == 2);
	CHECK(boost::source(*(++in_i), *grid) == 4);
	CHECK(++in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 8);
	CHECK(edge_id[*out_i] == 0);
	CHECK(++out_i == out_end);

	v = boost::vertex(8, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::source(*in_i, *grid) == 5);
	CHECK(boost::source(*(++in_i), *grid) == 7);
	CHECK(++in_i == in_end);
	CHECK(out_i == out_end);
      }
      SECTION("SENW") {
	DPAG* grid = instance->orientation(1);
	EdgeId edge_id = boost::get(boost::edge_index, *grid);

	Vertex_d v = boost::vertex(0, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(out_i == out_end);
	CHECK(boost::source(*in_i, *grid) == 3);
	CHECK(boost::source(*(++in_i), *grid) == 1);
	CHECK(++in_i == in_end);

	v = boost::vertex(5, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::source(*in_i, *grid) == 8);
	CHECK(++in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 4);
	CHECK(edge_id[*out_i] == 0);
	CHECK(boost::target(*(++out_i), *grid) == 2);
	CHECK(++out_i == out_end);

	v = boost::vertex(8, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::target(*out_i, *grid) == 7);
	CHECK(edge_id[*out_i] == 0);
	CHECK(boost::target(*(++out_i), *grid) == 5);
	CHECK(edge_id[*out_i] == 1);
	CHECK(++out_i == out_end);
	CHECK(in_i == in_end);
      }
      SECTION("NESW") {
	DPAG* grid = instance->orientation(2);
	EdgeId edge_id = boost::get(boost::edge_index, *grid);

	Vertex_d v = boost::vertex(2, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 1);
	CHECK(edge_id[*out_i] == 0);
	CHECK(boost::target(*(++out_i), *grid) == 5);
	CHECK(edge_id[*out_i] == 1);
	CHECK(++out_i == out_end);

	v = boost::vertex(3, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::source(*in_i, *grid) == 0);
	CHECK(boost::source(*(++in_i), *grid) == 4);
	CHECK(++in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 6);
	CHECK(edge_id[*out_i] == 0);
	CHECK(++out_i == out_end);

	v = boost::vertex(8, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::source(*in_i, *grid) == 5);
	CHECK(++in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 7);
	CHECK(edge_id[*out_i] == 0);
	CHECK(++out_i == out_end);
      }
      SECTION("SWNE") {
	DPAG* grid = instance->orientation(3);
	EdgeId edge_id = boost::get(boost::edge_index, *grid);

	Vertex_d v = boost::vertex(6, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 7);
	CHECK(edge_id[*out_i] == 0);
	CHECK(boost::target(*(++out_i), *grid) == 3);
	CHECK(edge_id[*out_i] == 1);
	CHECK(++out_i == out_end);

	v = boost::vertex(5, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(boost::source(*in_i, *grid) == 8);
	CHECK(boost::source(*(++in_i), *grid) == 4);
	CHECK(++in_i == in_end);
	CHECK(boost::target(*out_i, *grid) == 2);
	CHECK(edge_id[*out_i] == 0);
	CHECK(++out_i == out_end);

	v = boost::vertex(2, *grid);
	boost::tie(out_i, out_end)=boost::out_edges(v, *grid);
	boost::tie(in_i, in_end)=boost::in_edges(v, *grid);
	CHECK(out_i == out_end);
	CHECK(boost::source(*in_i, *grid) == 5);
	CHECK(boost::source(*(++in_i), *grid) == 1);
	CHECK(++in_i == in_end);
      }
    }

    SECTION("topological sort") {
      SECTION("NWSE") {
	vector<int> top_sort = instance->topological_order(0);
	CHECK(top_sort.size() == instance->num_nodes());

	CHECK(top_sort[0] == 0);
	CHECK(top_sort[1] == 3);
	CHECK(top_sort[5] == 7);
	CHECK(top_sort[instance->num_nodes()-1] == instance->num_nodes()-1);
      }
      SECTION("SENW") {
	vector<int> top_sort = instance->topological_order(1);
	CHECK(top_sort.size() == instance->num_nodes());

	CHECK(top_sort[0] == 8);
	CHECK(top_sort[1] == 7);
	CHECK(top_sort[5] == 3);
	CHECK(top_sort[instance->num_nodes()-1] == 0);
      }
      SECTION("NESW") {
	vector<int> top_sort = instance->topological_order(2);
	CHECK(top_sort.size() == instance->num_nodes());

	CHECK(top_sort[0] == 2);
	CHECK(top_sort[1] == 5);
	CHECK(top_sort[5] == 7);
	CHECK(top_sort[instance->num_nodes()-1] == 6);
      }
      SECTION("SWNE") {
	vector<int> top_sort = instance->topological_order(3);
	CHECK(top_sort.size() == instance->num_nodes());

	CHECK(top_sort[0] == 6);
	CHECK(top_sort[1] == 7);
	CHECK(top_sort[5] == 5);
	CHECK(top_sort[instance->num_nodes()-1] == 2);
      }
    }
  }
}
