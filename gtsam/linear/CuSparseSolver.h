/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/linear/Scatter.h>
#include <gtsam/inference/Ordering.h>

#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <cusolverSp.h>
#include <cusparse_v2.h>

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
  void allocate(int n, int nnz);
  void free();

  cusolverSpHandle_t cusolverH_ = nullptr;
  cusparseMatDescr_t descrA_ = nullptr;

  // Unified memory buffers (Jetson zero-copy)
  int* csrRowPtr_ = nullptr;
  int* csrColInd_ = nullptr;
  double* csrVal_ = nullptr;
  double* b_ = nullptr;
  double* x_ = nullptr;

  int allocN_ = 0;
  int allocNnz_ = 0;

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm_;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permInv_;
};

VectorValues cuSparseSolve(const GaussianFactorGraph& gfg,
                           const Ordering& ordering,
                           const Scatter& scatter);

}  // namespace gtsam
