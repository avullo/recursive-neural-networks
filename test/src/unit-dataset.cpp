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

#include "catch.hpp"

#include "General.h"
#include "Options.h"
#include "DataSet.h"
#include <sstream>
using namespace std;


TEST_CASE("Basic dataset tests", "[dataset]") {
  // prepare arguments and read configuration file
  setenv("RNNOPTIONTYPE", "train", 1);
  char* argv[] = { (char*)"dummy", (char*)"-c", (char*)"data/rnn.conf" };
  Options::instance()->parse_args(3, argv);

  // vertexIt v_i, v_end;
  // outIter out_i, out_end;
  // ieIter in_i, in_end;

  Options::instance()->domain(DOAG);
  DataSet ds("data/dataset.gph");  
  CHECK(ds.size() == 3);
  int count = 0;
  for(DataSet::iterator it=ds.begin(); it!=ds.end(); ++it) {
    Instance* instance = *it;
    string got = instance->id();
    stringstream ss; ss << ++count;
    string expected = "dummy" + ss.str();
    CHECK(got == expected);
    
    CHECK(instance->domain() == DOAG);
    CHECK(instance->num_nodes() == 4);
    Node* n = instance->node(1);
    vector<float> node_input = n->input(), node_target = n->target();
    CHECK(node_input.size() == 3);
    CHECK(node_input.size() == 3);
    CHECK(node_input[0] == .4f); CHECK(node_input[1] == .5f); CHECK(node_input[2] == .6f);

    CHECK(instance->num_orient() == 1);
    
    DPAG* doag = instance->orientation(0);
    int V = boost::num_vertices(*doag);
    CHECK(V == instance->num_nodes());

    VertexId vertex_id;
    vertexIt v_i, v_end;
    outIter out_i, out_end;
    ieIter in_i, in_end;
    EdgeId edge_id = boost::get(boost::edge_index, *doag);
    
    if(got == "dummy1") {
      CHECK(node_target[0] == .6f); CHECK(node_target[1] == .5f); CHECK(node_target[2] == .4f);

      Vertex_d v = boost::vertex(0, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 1);
      CHECK(edge_id[*out_i] == 0);
      CHECK(boost::target(*(++out_i), *doag) == 2);
      CHECK(edge_id[*out_i] == 1);
      CHECK(boost::target(*(++out_i), *doag) == 3);
      CHECK(edge_id[*out_i] == 2);
      CHECK(++out_i == out_end);

      v = boost::vertex(1, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 0);
      CHECK(++in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 2);
      CHECK(edge_id[*out_i] == 0);
      CHECK(++out_i == out_end);

      v = boost::vertex(2, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 0);
      CHECK(boost::source(*(++in_i), *doag) == 1);      
      CHECK(++in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 3);
      CHECK(edge_id[*out_i] == 0);      
      CHECK(++out_i == out_end);

      v = boost::vertex(3, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 0);
      CHECK(boost::source(*(++in_i), *doag) == 2);
      CHECK(++in_i == in_end);
      CHECK(out_i == out_end);
    } else if(got == "dummy2") {
      CHECK(node_target[0] == .9f); CHECK(node_target[1] == .8f); CHECK(node_target[2] == .7f);

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
    } else {
      CHECK(node_target[0] == .3f); CHECK(node_target[1] == .2f); CHECK(node_target[2] == .1f);
      
      Vertex_d v = boost::vertex(0, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(in_i == in_end);
      CHECK(boost::target(*out_i, *doag) == 1);
      CHECK(edge_id[*out_i] == 0);
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
      CHECK(boost::target(*out_i, *doag) == 3);
      CHECK(edge_id[*out_i] == 0);      
      CHECK(++out_i == out_end);

      v = boost::vertex(3, *doag);
      boost::tie(out_i, out_end)=boost::out_edges(v, *doag);
      boost::tie(in_i, in_end)=boost::in_edges(v, *doag);
      CHECK(boost::source(*in_i, *doag) == 1);
      CHECK(boost::source(*(++in_i), *doag) == 2);
      CHECK(++in_i == in_end);
      CHECK(out_i == out_end);
    }
  }
}
