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

#ifndef MCL_ARMIJO_H
#define MCL_ARMIJO_H

#include "Problem.hpp"

namespace mcl {
namespace optlib {

// Backtracking-Armijo with optional curvature (weak wolfe II) interpolation
template<typename Scalar, int DIM, typename P>
class Armijo {
public:
	typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatrixX;

	static Scalar linesearch(const VectorX &x, const VectorX &p, P &problem, Scalar alpha_init, bool curvature=false) {

		const Scalar tau = 0.7; // used if curvature = false
		const Scalar beta = 1e-5;
		const int max_iter = 1000000;
		Scalar alpha = std::abs(alpha_init);
		VectorX grad;
		if( DIM == Eigen::Dynamic ){ grad = VectorX::Zero(x.rows()); }
		Scalar fx0 = problem.gradient(x, grad);
		Scalar gtp = grad.dot(p);
		Scalar fxp = fx0;
		Scalar alphap = alpha;

		int iter = 0;
		for( ; iter < max_iter; ++iter ){
			Scalar fxa = problem.value(x + alpha*p);
			Scalar fx0_fxa = fx0 + alpha*beta*gtp; // Armijo condition I
			if( fxa <= fx0_fxa ){ break; } // sufficient decrease

			// Use interpolation?
			if( curvature ){
				Scalar alpha_tmp = iter == 0 ?
					( gtp / (2.0 * (fx0 + gtp - fxa)) ) :
					cubic( fx0, gtp, fxa, alpha, fxp, alphap );
				fxp = fxa;
				alphap = alpha;
				alpha = range( alpha_tmp, 0.1*alpha, 0.5*alpha );
			} else {
				alpha *= tau;
			}
		}

		if( iter == max_iter ){
			printf("Armijo::linesearch Error: Reached max_iters\n");
			return -1;
		}

		return alpha;
	}

private:

	static Scalar range( Scalar alpha, Scalar low, Scalar high ){
		if( alpha < low ){ return low; }
		else if( alpha > high ){ return high; }
		return alpha;
	}

	// Cubic interpolation
	// fx0 = f(x0)
	// gtp = f'(x0)^T p
	// fxa = f(x0 + alpha*p)
	// alpha = step length
	// fxp = previous fxa
	// alphap = previous alpha
	static Scalar cubic( Scalar fx0, Scalar gtp, Scalar fxa, Scalar alpha, Scalar fxp, Scalar alphap ){
		typedef Eigen::Matrix<Scalar,2,1> Vec2;
		typedef Eigen::Matrix<Scalar,2,2> Mat2;

		Scalar mult = 1.0 / ( alpha*alpha * alphap*alphap * (alpha-alphap) );
		Mat2 A;
		A(0,0) = alphap*alphap;
		A(0,1) = -alpha*alpha;
		A(1,0) = -alphap*alphap*alphap;
		A(1,1) = alpha*alpha*alpha;	
		Vec2 B;
		B[0] = fxa - fx0 - alpha*gtp;		
		B[1] = fxp - fx0 - alphap*gtp;
		Vec2 r = mult * A * B;
		if( std::abs(r[0]) <= 0.0 ){ return -gtp / (2.0*r[1]); } // if quadratic
		Scalar d = std::sqrt( r[1]*r[1] - 3.0*r[0]*gtp ); // discriminant
		return (-r[1] + d) / (3.0*r[0]);
	}

}; // end class Armijo

} // ns optlib
} // ns mcl

#endif
