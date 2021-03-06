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

#ifndef _STRUCTURED_DOMAIN_H_
#define _STRUCTURED_DOMAIN_H_

#include <stdexcept>

/*
 *
 * Define aspects related to the representation of structured domains.
 *
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
int num_orientations(Domain domain)
  throw(std::logic_error);
  
typedef enum Transduction {
  SUPER_SOURCE = 0,
  IO_ISOMORPH
} Transduction;

/*
 * Types of learning problems on a structured domain
 */
typedef enum {
  UNDEFINED   = 1<<0,
  REGRESSION  = 1<<1,
  BINARYCLASS = 1<<2,
  MULTICLASS  = 1<<3,
} Problem;

#endif // _STRUCTURED_DOMAIN_H_
