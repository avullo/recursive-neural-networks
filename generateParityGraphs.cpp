#include "require.h"
#include "General.h"
#include "DPAG.h"
#include <cstdlib>
#include <ctime>
#include <set>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

/*
  This program allow to generate and store to file a set of random graphs 
  where each node has a target output label. Main purpose is to test
  forward-backward phase of my RecursiveNN with io-isomorph structural
  trasductions.
*/

void generateRandomParityGraph(int index, int min_nodes, int max_nodes, int max_outdegree, ostream& os = cout) {
  // generate a random number of nodes (between min_nodes and max_nodes)
  int num_nodes = min_nodes + (int)((double)(max_nodes-min_nodes) * rand()/(RAND_MAX+1.0));
  os << "graph" << index << " " << num_nodes << endl << endl;

  // Initialize two graphs DG RG with num_nodes
  DPAG DG(num_nodes), RG(num_nodes);
  vector<vector<bool> > nodes_states_label(num_nodes, vector<bool>(4, false));

  Vertex_d currentNode;
  outIter out_i, out_end;

  for(int i=0; i<num_nodes; i++) {
    // Assign random boolean label to current node
    nodes_states_label[i][2] = (rand()/(RAND_MAX+1.0) > .5)?true:false;

    // Generate a random number of children (0<=od<=max_outdegree)
    // where child index j has interval i+1<=j<=num_nodes-1. 
    // Control to avoid self edges and duplicates.
    set<int> children;
    int num_children = (int)((double)max_outdegree * rand()/(RAND_MAX+1.0));
    if(num_children >= num_nodes - i)
      num_children = num_nodes-i-1;
    
    int j = 0;
    while(j<num_children) {
      int child  = i + (int)((double)(num_nodes-i) * rand()/(RAND_MAX+1.0));
      if(!(child == i)&&(children.find(child)==children.end())) {
	boost::add_edge(i, child, EdgeProperty(0), DG); 
	boost::add_edge(child, i, EdgeProperty(0), RG);
	children.insert(child);
	j++;
      }
    }
  }

  // Get for each node its forward and backward states, 
  for(int i=num_nodes-1; i>=0; i--) {
    currentNode = boost::vertex(i, DG);
    int num_bits_at_1 = 0;
    for(boost::tie(out_i, out_end)=out_edges(currentNode, DG); 
	out_i!=out_end; ++out_i) {
      if(nodes_states_label[target(*out_i, DG)][0])
	num_bits_at_1++;
    }
    if(nodes_states_label[i][2])
      num_bits_at_1++;
    if(num_bits_at_1%2)
      nodes_states_label[i][0] = true;
  }

  for(int i=0; i<num_nodes; i++) {
    currentNode = boost::vertex(i, RG);
    int num_bits_at_1 = 0;
    for(boost::tie(out_i, out_end)=out_edges(currentNode, RG); 
	out_i!=out_end; ++out_i) {
      if(nodes_states_label[target(*out_i, RG)][1])
	num_bits_at_1++;
    }
    if(nodes_states_label[i][2])
      num_bits_at_1++;
    if(num_bits_at_1%2)
      nodes_states_label[i][1] = true;
  }

  // write each node input/output
  for(int i=0; i<num_nodes; i++) {
    // compute output (target) label based on node input
    // and forward-backward states
    int num_bits_at_1 = 0;
    for(int j=0; j<3; j++)
      if(nodes_states_label[i][j])
	num_bits_at_1++;
    if(num_bits_at_1%2)
      nodes_states_label[i][3] = true;
    
    os << nodes_states_label[i][2] << " " // input
       << nodes_states_label[i][0] << " " // input
       << nodes_states_label[i][1] << " " // input
       << nodes_states_label[i][3] // target
       << endl;
  }

  // write skeleton
  os << endl;
  for(int i=0; i<num_nodes; i++) {
    os << i << ' ';
    currentNode = boost::vertex(i, DG);
    for(boost::tie(out_i, out_end)=out_edges(currentNode, DG); 
	out_i!=out_end; ++out_i) {
      os << target(*out_i, DG) << " ";
    }
    os << endl;
  }

  os << endl << endl;
  //boost::print_graph(DG, boost::get(boost::vertex_index,DG));
  //boost::print_graph(RG, boost::get(boost::vertex_index,RG));
}

int main(int argc, char* argv[]) {
  // check command line arguments
  requireArgs(argc, 5,
	      "Usage:\n\tgenerateParityGraphs <number of graphs> <min #nodes> <max #nodes> <max outdegree> <filename>\n");
  
  int num_graphs = atoi(argv[1]);
  int min_nodes = atoi(argv[2]);
  int max_nodes = atoi(argv[3]);
  int max_outdegree = atoi(argv[4]);
 
  // open an output stream with provided filename
  ofstream os(argv[5]);
  assure(os, argv[5]);

  //int seed =20010601;
  // initialize random seed
  srand(time(0));

  os << num_graphs << endl << endl;
  
  for(int i=1; i<=num_graphs; ++i)
    generateRandomParityGraph(i, min_nodes, max_nodes, max_outdegree, os);
}
