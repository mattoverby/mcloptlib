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

#ifndef MCL_MINIMIZER_H
#define MCL_MINIMIZER_H

#include "Problem.hpp"
#include "Backtracking.hpp"
#include "MoreThuente.hpp"
#include "WolfeBisection.hpp"
#include <memory>

namespace mcl {
namespace optlib {

template<typename Scalar, int DIM>
class Minimizer {
public:
	typedef Eigen::Matrix<Scalar,DIM,1> VecX;
	static const int FAILURE = -1; // returned by minimize if an error is encountered

	struct Settings {
		int verbose;
		int max_iters; // usually set by derived constructors
		LinesearchMethod ls_method; // see LinesearchMethod
		Settings() : verbose(0), max_iters(100),
			ls_method(LinesearchMethod::Backtracking) {}
	} m_settings;

	//
	// Performs optimization
	//
	virtual int minimize(Problem<Scalar,DIM> &problem, VecX &x) = 0;


protected:
	// Helper function for creating the linesearch object
	void make_linesearch( std::unique_ptr< Linesearch<Scalar,DIM> > &ptr ){
		typedef std::unique_ptr< Linesearch<Scalar,DIM> > LSPtr;
		switch( m_settings.ls_method ){
			default: { ptr = LSPtr( new Backtracking<Scalar,DIM>() ); } break;
			case LinesearchMethod::None: { ptr = nullptr; } break;
			case LinesearchMethod::MoreThuente: { ptr = LSPtr( new MoreThuente<Scalar,DIM>() ); } break;
			case LinesearchMethod::Backtracking: { ptr = LSPtr( new Backtracking<Scalar,DIM>() ); } break;
			case LinesearchMethod::BacktrackingCurvature: { ptr = LSPtr( new BacktrackingCurvature<Scalar,DIM>() ); } break;
			case LinesearchMethod::WeakWolfeBisection: { ptr = LSPtr( new WolfeBisection<Scalar,DIM>() ); } break;
		}
	}
};

} // ns optlib
} // ns mcl

#endif
