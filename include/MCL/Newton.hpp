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

#ifndef MCL_NEWTON_H
#define MCL_NEWTON_H

#include "Armijo.hpp"

namespace mcl {
namespace optlib {

template<typename Scalar, int DIM>
class Newton {
private:
	typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatrixX;

public:
	int max_iters;
	Scalar eps;
	bool use_linesearch;

	struct Init {
		int max_iters;
		Scalar eps; // 0 = run full iterations
		bool use_linesearch;
		Init() : max_iters(20), eps(0), use_linesearch(true) {}
	};

	Newton( const Init &init = Init() ) : max_iters(init.max_iters),
		eps(init.eps), use_linesearch(init.use_linesearch) {}

	int minimize(Problem<Scalar,DIM> &problem, VectorX &x){

		VectorX grad, delta_x;
		MatrixX hess;
		if( DIM  == Eigen::Dynamic ){
			int dim = x.rows();
			grad.resize(dim);
			hess.resize(dim,dim);
			delta_x.resize(dim);
		}

		int iter = 0;
		for( ; iter < max_iters; ++iter ){

			problem.gradient(x,grad);
			problem.hessian(x,hess);

			if( DIM == Eigen::Dynamic || DIM > 4 ){
				delta_x = hess.householderQr().solve(-grad);
			} else {
				delta_x = -hess.inverse()*grad;
			}

			Scalar alpha = !use_linesearch ? Scalar(1) :
				Armijo<Scalar, DIM, decltype(problem)>::linesearch(x, delta_x, problem, 1);
			x += alpha * delta_x;
			if( alpha * delta_x.squaredNorm() <= eps ){ break; }
		}

		return iter;
	}

// Copied from https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
//	Method			Requirements	Spd (sm)	Spd (lg)	Accuracy
//	partialPivLu()		Invertible	++		++		+
//	fullPivLu()		None		-		- -		+++
//	householderQr()		None		++		++		+
//	colPivHouseholderQr()	None		++		-		+++
//	fullPivHouseholderQr()	None		-		- -		+++
//	llt()			PD		+++		+++		+
//	ldlt()			P/N SD 		+++		+		++

};

} // ns optlib
} // ns mcl

#endif
