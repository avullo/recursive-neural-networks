#ifndef _STRUCTURED_DOMAIN_H_
#define _STRUCTURED_DOMAIN_H_

#include "DPAG.h"
#include "Node.h"
#include <cassert>
#include <vector>

/*

  Define aspects related to the representation of structured domains.
  
*/

/*
  List supported structured domain types:

  - DOAG: general DOAG
  - SEQUENCE, serial order sequence, i.e. left-to-right sequence
  - LINEARCHAIN, unordered chain, i.e. bidirectional sequence processing
  - NARYTREE, n-ary tree
  - UG, undirected graph, i.e. multiple-orientations processing, assume serial order defined on vertex indices
  - GRID2D, two-dimensional grid, no orientation

*/
typedef enum Domain {
  DOAG = 0, 
  SEQUENCE,
  LINEARCHAIN,
  NARYTREE,
  UG, 
  GRID2D, 
} Domain;

// Return the number of orientations the RNN must consider
// to process an instance in the given domain
int num_orientations(Domain domain) {
  switch(domain) {
  case DOAG: return 1;
  case SEQUENCE: return 1;
  case LINEARCHAIN: return 2;
  case NARYTREE: return 1;
  case UG: return 2;
  case GRID2D: return 4;
  default: // TODO raise exception
  }
}

typedef enum Transduction {
  SUPER_SOURCE = 0,
  IO_ISOMORPH
} Transduction;


#endif // _STRUCTURED_DOMAIN_H_
