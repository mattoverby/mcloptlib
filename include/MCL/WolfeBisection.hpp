// The MIT License (MIT)
// Copyright (c) 2018 Matt Overby
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

#ifndef MCL_WOLFEBISECTION_H
#define MCL_WOLFEBISECTION_H

#include "Problem.hpp"

namespace mcl {
namespace optlib {

// Bisection method for Weak Wolfe conditions
template<typename Scalar, int DIM>
class WolfeBisection {
public:
	typedef Eigen::Matrix<Scalar,DIM,1> VecX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatX;

	static inline Scalar search(const VecX &x, const VecX &p, Problem<Scalar,DIM> &problem, Scalar alpha0) {

		const Scalar c1 = 0.3;
		const Scalar c2 = 0.6;
		const int max_iter = 100000;
		const int dim = x.rows();
		const Scalar minChange = 1e-10;

		Scalar low = 0.0;
		Scalar high = -1.0; // set when needed
		Scalar alpha = alpha0;
		Scalar alpha_last = 0.0;

		VecX grad0, grad_new;
		if( DIM == Eigen::Dynamic ){
			grad0 = VecX::Zero(dim);
			grad_new = VecX::Zero(dim);
		}
		Scalar fx = problem.gradient(x, grad0);
		const Scalar gtp = grad0.dot(p);

		int iter = 0;
		for( ; iter < max_iter; ++iter ){

			// Value and gradient at current step length
			grad_new.setZero();
			Scalar fx_new = problem.gradient(x + alpha*p, grad_new);

			if( fx_new > fx + c1*alpha*gtp ){
				high = alpha;
				alpha = 0.5 * ( high + low );
			}
			else if( grad_new.dot(p) < c2*gtp ){
				low = alpha;
				if( high < 0 ){ alpha = 2.0 * low; }
				else{ alpha = 0.5 * ( high + low ); }
			}
			else{ break; }

			// Exit if change in bisection step too low.
			// Not perfectly safe, but probably fine?
			if( std::abs(alpha-alpha_last) < minChange ){ break; }
			alpha_last = alpha;
		}

		if( iter == max_iter ){
			printf("Backtracking::linesearch Error: Reached max_iters\n");
			return -1;
		}

		return alpha;

	}
};

}
}

#endif
