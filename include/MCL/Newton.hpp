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
#include "MoreThuente.hpp"
#include "Minimizer.hpp"

namespace mcl {
namespace optlib {

template<typename Scalar, int DIM>
class Newton : public Minimizer<Scalar,DIM> {
private:
	typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatrixX;

public:
	int max_iters;
	Newton() : max_iters(20) {}
	void set_max_iters( int iters ){ max_iters = iters; }
	void set_verbose( int v ){ (void)(v); } // TODO

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

			Scalar rate =
				Armijo<Scalar, DIM, decltype(problem)>::linesearch(x, delta_x, problem, 1);
//				MoreThuente<Scalar, DIM, decltype(problem)>::linesearch(x, delta_x, problem, 1);

			if( rate <= 0 ){
				printf("Newton::minimize: Failure in linesearch");
				return Minimizer<Scalar,DIM>::FAILURE;
			}

			x += rate * delta_x;
			if( problem.converged(x,grad) ){ break; }
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
