// The MIT License (MIT)
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

#ifndef MCL_PROBLEM_H
#define MCL_PROBLEM_H

#if MCL_DEBUG == 1
#include <iostream>
#endif

#include <Eigen/Dense>

namespace mcl {
namespace optimize {

template<typename Scalar, int DIM>
class Problem {
private:
	typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatrixX;

public:
	// Compute objective value
	virtual Scalar value(const VectorX &x) = 0;

	// Compute gradient
	virtual void gradient(const VectorX &x, VectorX &grad){
		finiteGradient(x, grad);
	}

	// Compute hessian
	virtual void hessian(const VectorX &x, MatrixX &hessian){
		finiteHessian(x, hessian);
	}

	// Compute value and gradient at the same time
	virtual Scalar value_gradient(const VectorX &x, VectorX &grad){
		gradient(x, grad);
		return value(x);
	}

	// Gradient with finite differences
	inline void finiteGradient(const VectorX &x, VectorX &grad){
		const int accuracy = 0; // accuracy can be 0, 1, 2, 3
		const Scalar eps = 2.2204e-6;
		const std::vector< std::vector<Scalar> > coeff =
		{ {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} };
		const std::vector< std::vector<Scalar> > coeff2 =
		{ {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} };
		const std::vector<Scalar> dd = {2, 12, 60, 840};
		VectorX finiteDiff(DIM);
		for(int d = 0; d < DIM; ++d){
			finiteDiff[d] = 0;
			for (int s = 0; s < 2*(accuracy+1); ++s){
				VectorX xx = x.eval();
				xx[d] += coeff2[accuracy][s]*eps;
				finiteDiff[d] += coeff[accuracy][s]*value(xx);
			}
			finiteDiff[d] /= (dd[accuracy]* eps);
		}
		grad = finiteDiff;
	} // end finite grad

	// Hessian with finite differences
	inline void finiteHessian(const VectorX &x, MatrixX &hess){
		const Scalar eps = std::numeric_limits<Scalar>::epsilon()*10e7;
		for(int i = 0; i < DIM; ++i){
			for(int j = 0; j < DIM; ++j){
				VectorX xx = x;
				Scalar f4 = value(xx);
				xx[i] += eps;
				xx[j] += eps;
				Scalar f1 = value(xx);
				xx[j] -= eps;
				Scalar f2 = value(xx);
				xx[j] += eps;
				xx[i] -= eps;
				Scalar f3 = value(xx);
				hess(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);
			}
		}
	} // end finite hess
};

}
}

#endif
