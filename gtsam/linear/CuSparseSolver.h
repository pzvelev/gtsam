/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <gtsam/config.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/linear/Scatter.h>
#include <gtsam/inference/Ordering.h>

#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <cusolverSp.h>
#include <cusparse.h>

#include <memory>

namespace gtsam {

class GTSAM_EXPORT CuSparseSolver {
 public:
  CuSparseSolver();
  ~CuSparseSolver();

  CuSparseSolver(const CuSparseSolver&) = delete;
  CuSparseSolver& operator=(const CuSparseSolver&) = delete;

  VectorValues solve(const GaussianFactorGraph& gfg,
                     const Ordering& ordering,
                     const Scatter& scatter);

 private:
  void freeAtA();

  cusparseHandle_t cusparseH_ = nullptr;
  cusolverSpHandle_t cusolverH_ = nullptr;
  cusparseMatDescr_t descrAtA_ = nullptr;

  // A'A result buffers (GPU managed memory)
  int* ataRowPtr_ = nullptr;
  int* ataColInd_ = nullptr;
  double* ataVal_ = nullptr;
  int64_t ataNnz_ = 0;

  // RHS and solution buffers
  double* rhs_ = nullptr;
  double* sol_ = nullptr;
  int allocRhsN_ = 0;
};

VectorValues cuSparseSolve(const GaussianFactorGraph& gfg,
                           const Ordering& ordering,
                           const Scatter& scatter);

}  // namespace gtsam
