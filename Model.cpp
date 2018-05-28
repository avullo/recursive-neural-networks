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

#include "Options.h"
#include "Model.h"
#include "RecursiveNN.h"
using namespace std;

/*
 * TODO
 *
 * introduce HA/OA/EMP as options and dynamically instantiate network
 * with the corresponding classes
 *
 */
Model* Model::factory(const string& netname)
  throw(BadModelCreation) {
  
  Problem problem = Options::instance()->problem();
  
  if(problem & BINARYCLASS) {
    if(netname == "")
      return new RecursiveNN<TanH, Sigmoid, MGradientDescent>();
    else
      return new RecursiveNN<TanH, Sigmoid, MGradientDescent>(netname.c_str());
    
  } else if(problem & (MULTICLASS | REGRESSION)) {
    if(netname == "")
      return new RecursiveNN<TanH, Linear, MGradientDescent>();
    else
      return new RecursiveNN<TanH, Linear, MGradientDescent>(netname.c_str());
    
  } else
    throw BadModelCreation("Unknown problem type");
}
