/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <gtsam/linear/CuSparseSolver.h>
#include <gtsam/linear/SparseEigen.h>

#include <cuda_runtime.h>

#include <stdexcept>

namespace gtsam {

namespace {

void checkCuda(cudaError_t status, const char* msg) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
}

void checkCusolver(cusolverStatus_t status, const char* msg) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(msg) + ": cusolver error " + std::to_string(status));
}

void checkCusparse(cusparseStatus_t status, const char* msg) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string(msg) + ": cusparse error " + std::to_string(status));
}

}  // namespace

CuSparseSolver::CuSparseSolver() {
  checkCusolver(cusolverSpCreate(&cusolverH_), "cusolverSpCreate");
  checkCusparse(cusparseCreateMatDescr(&descrA_), "cusparseCreateMatDescr");
  cusparseSetMatType(descrA_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA_, CUSPARSE_INDEX_BASE_ZERO);
}

CuSparseSolver::~CuSparseSolver() {
  free();
  if (descrA_) cusparseDestroyMatDescr(descrA_);
  if (cusolverH_) cusolverSpDestroy(cusolverH_);
}

void CuSparseSolver::allocate(int n, int nnz) {
  if (n <= allocN_ && nnz <= allocNnz_) return;

  free();

  checkCuda(cudaMallocManaged(&csrRowPtr_, sizeof(int) * (n + 1)), "malloc csrRowPtr");
  checkCuda(cudaMallocManaged(&csrColInd_, sizeof(int) * nnz), "malloc csrColInd");
  checkCuda(cudaMallocManaged(&csrVal_, sizeof(double) * nnz), "malloc csrVal");
  checkCuda(cudaMallocManaged(&b_, sizeof(double) * n), "malloc b");
  checkCuda(cudaMallocManaged(&x_, sizeof(double) * n), "malloc x");

  allocN_ = n;
  allocNnz_ = nnz;
}

void CuSparseSolver::free() {
  if (csrRowPtr_) { cudaFree(csrRowPtr_); csrRowPtr_ = nullptr; }
  if (csrColInd_) { cudaFree(csrColInd_); csrColInd_ = nullptr; }
  if (csrVal_) { cudaFree(csrVal_); csrVal_ = nullptr; }
  if (b_) { cudaFree(b_); b_ = nullptr; }
  if (x_) { cudaFree(x_); x_ = nullptr; }
  allocN_ = 0;
  allocNnz_ = 0;
}

VectorValues CuSparseSolver::solve(const GaussianFactorGraph& gfg,
                                    const Ordering& ordering,
                                    const Scatter& scatter) {
  // Get augmented sparse Jacobian [A | b] as Eigen sparse matrix
  SparseEigen Ab = sparseJacobianEigen(gfg, ordering);

  const int n = Ab.cols() - 1;  // last column is RHS

  // Extract A (all columns except last) and b (last column)
  // Form normal equations: AtA = A'A, Atb = A'b
  SparseEigen A = Ab.leftCols(n);
  Eigen::VectorXd rhs = Eigen::VectorXd(Ab.col(n));
  Eigen::VectorXd Atb = A.transpose() * rhs;

  // Form A'A (SPD). Store upper triangle in CSC = lower triangle in CSR,
  // which is what cusolverSpDcsrlsvchol reads.
  SparseEigen AtA = (A.transpose() * A).triangularView<Eigen::Upper>();
  AtA.makeCompressed();

  // AMD ordering for fill reduction (recompute every iteration —
  // sparsity pattern changes as factors/variables are added)
  Eigen::AMDOrdering<int> amdOrdering;
  amdOrdering(AtA.selfadjointView<Eigen::Upper>(), perm_);
  permInv_ = perm_.inverse();

  // Apply permutation via selfadjointView to correctly handle single-triangle input
  SparseEigen AtA_perm(n, n);
  AtA_perm.selfadjointView<Eigen::Upper>() =
      AtA.selfadjointView<Eigen::Upper>().twistedBy(perm_);
  AtA_perm.makeCompressed();
  Eigen::VectorXd Atb_perm = perm_ * Atb;

  const int nnz = AtA_perm.nonZeros();

  // Allocate unified memory buffers
  allocate(n, nnz);

  // Eigen CSC is equivalent to CSR for symmetric matrices
  // CSC outer = CSR row pointers, CSC inner = CSR column indices
  std::memcpy(csrRowPtr_, AtA_perm.outerIndexPtr(), sizeof(int) * (n + 1));
  std::memcpy(csrColInd_, AtA_perm.innerIndexPtr(), sizeof(int) * nnz);
  std::memcpy(csrVal_, AtA_perm.valuePtr(), sizeof(double) * nnz);
  std::memcpy(b_, Atb_perm.data(), sizeof(double) * n);

  // Ensure data is visible to GPU (unified memory coherence)
  cudaDeviceSynchronize();

  int singularity = 0;
  checkCusolver(
      cusolverSpDcsrlsvchol(
          cusolverH_, n, nnz, descrA_,
          csrVal_, csrRowPtr_, csrColInd_,
          b_, 0.0, 0, x_, &singularity),
      "cusolverSpDcsrlsvchol");

  cudaDeviceSynchronize();

  if (singularity != -1)
    throw std::runtime_error("CuSparseSolver: matrix is singular at row " +
                             std::to_string(singularity));

  // Apply inverse permutation and construct VectorValues
  Eigen::Map<Eigen::VectorXd> x_perm(x_, n);
  Eigen::VectorXd solution = permInv_ * x_perm;

  return VectorValues(solution, scatter);
}

// Thread-local solver instance for reuse across optimizer iterations
static thread_local CuSparseSolver tlsSolver;

VectorValues cuSparseSolve(const GaussianFactorGraph& gfg,
                           const Ordering& ordering,
                           const Scatter& scatter) {
  return tlsSolver.solve(gfg, ordering, scatter);
}

}  // namespace gtsam
