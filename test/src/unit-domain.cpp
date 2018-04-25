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

#include "StructuredDomain.h"
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

TEST_CASE("Basic facs about a domain", "[structured domain]") {
  Domain domains[] = { DOAG, SEQUENCE, LINEARCHAIN, NARYTREE, UG, GRID2D };
  int orientations[] = { 1, 1, 2, 1, 2, 4 };

  for(int i=0; i<6; ++i)
    CHECK(num_orientations(domains[i]) == orientations[i]);
  CHECK_THROWS(num_orientations((Domain)-1));
  CHECK_THROWS(num_orientations((Domain)6));

}
