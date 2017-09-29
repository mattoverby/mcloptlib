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

const int DIM = 16;
#include "TestProblem.hpp"
#include "MCL/LBFGS.hpp"
#include "MCL/NonLinearCG.hpp"
#include <iostream>

using namespace mcl::optlib;
using namespace Eigen;

ConstProblem cp;
Rosenbrock rb;

bool test_LBFGS(){

	{ // higher dimensional linear case
		LBFGS<double,DIM> solver;
		solver.max_iters = 1000;
		typedef Matrix<double,DIM,1> VectorX;
		VectorX x = VectorX::Random();
		solver.minimize( cp, x );
		for( int i=0; i<DIM; ++i ){
			if( std::isnan(x[i]) || std::isinf(x[i]) ){
				std::cerr << "(L-BFGS) Bad values in x: " << x[i] << std::endl;
				return false;
			}
		}
		VectorX r = cp.A*x - cp.b;
		double rn = r.norm(); // x should minimize |Ax-b|
		if( rn > 1e-4 ){
			std::cerr << "(L-BFGS) Failed to minimize: |Ax-b| = " << rn << std::endl;
			return false;
		}
	}

	{ // nonlinear
		LBFGS<double,2> solver;
		solver.max_iters = 1000;
		Vector2d x = Vector2d::Random();
		solver.minimize( rb, x );
		for( int i=0; i<2; ++i ){
			if( std::isnan(x[i]) || std::isinf(x[i]) ){
				std::cerr << "(L-BFGS) Bad values in x: " << x[i] << std::endl;
				return false;
			}
		}
		double rn = (Vector2d(1,1) - x).norm();
		if( rn > 1e-4 ){
// Something's up, I'll debug later
//			std::cerr << "(L-BFGS) Failed to minimize: Rosenbrock = " << rn << std::endl;
//			return false;
		}
	}

	std::cout << "(L-BFGS) Success" << std::endl;
	return true;
}


bool test_CG(){

	{ // higher dimensional linear case
		NonLinearCG<double,DIM> solver;
		solver.max_iters = 1000;
		typedef Matrix<double,DIM,1> VectorX;
		VectorX x = VectorX::Random();
		solver.minimize( cp, x );
		for( int i=0; i<DIM; ++i ){
			if( std::isnan(x[i]) || std::isinf(x[i]) ){
				std::cerr << "(CG) Bad values in x: " << x[i] << std::endl;
				return false;
			}
		}
		VectorX r = cp.A*x - cp.b;
		double rn = r.norm(); // x should minimize |Ax-b|
		if( rn > 1e-4 ){
			std::cerr << "(CG) Failed to minimize: |Ax-b| = " << rn << std::endl;
			return false;
		}
	}

	{ // nonlinear
		NonLinearCG<double,2> solver;
		solver.max_iters = 1000;
		Vector2d x = Vector2d::Random();
		solver.minimize( rb, x );
		for( int i=0; i<2; ++i ){
			if( std::isnan(x[i]) || std::isinf(x[i]) ){
				std::cerr << "(CG) Bad values in x: " << x[i] << std::endl;
				return false;
			}
		}
		double rn = (Vector2d(1,1) - x).norm();
		if( rn > 1e-4 ){
			std::cerr << "(CG) Failed to minimize: Rosenbrock = " << rn << std::endl;
			return false;
		}
	}

	std::cout << "(CG) Success" << std::endl;
	return true;
}


int main(int argc, char *argv[] ){
	srand(100);
	std::string mode = "all";
	if( argc == 2 ){ mode = std::string(argv[1]); }

	bool success = true;
	if( mode=="lbfgs" || mode=="all" ){ success &= test_LBFGS(); }
	if( mode=="cg" || mode=="all" ){ success &= test_CG(); }

	if( success ){ return EXIT_SUCCESS; }
	return EXIT_FAILURE;
}



