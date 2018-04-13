// The MIT License (MIT)
// Copyright (c) 2017 Matt Overby
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MCL_LINESEARCH_H
#define MCL_LINESEARCH_H

#include "Problem.hpp"

namespace mcl {
namespace optlib {

// The different line search methods currently implemented
enum class LinesearchMethod {
	None = 0, // use step length = 1, not recommended, like, ever.
	MoreThuente, // TODO test this one for correctness
	Backtracking, // basic backtracking with sufficient decrease
	BacktrackingCurvature, // backtracking with cubic interpolation
	WeakWolfeBisection // slow
};

template<typename Scalar, int DIM>
class Linesearch {
public:
	typedef Eigen::Matrix<Scalar,DIM,1> VecX;
	static const int FAILURE = -1; // returned by line if an error is encountered

	struct Settings {
		Scalar sufficientDecrease; // sufficient decrease (Armijo)
		int max_iters;
		Settings() : sufficientDecrease(1e-4), max_iters(1e6) {}
	} m_settings;

	// Perform line search with:
	// x = opt variable
	// p = descent direction
	// problem = optimization function
	// alpha0 = initial line search guess
	virtual Scalar search(const VecX &x, const VecX &p, Problem<Scalar,DIM> &problem, Scalar alpha0) = 0;
	
};

} // ns optlib
} // ns mcl

#endif
