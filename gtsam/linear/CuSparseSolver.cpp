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

#include <cstring>
#include <memory>
#include <stdexcept>

namespace gtsam {

namespace {

void checkCuda(cudaError_t s, const char* msg) {
  if (s != cudaSuccess)
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(s));
}

void checkCusparse(cusparseStatus_t s, const char* msg) {
  if (s != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string(msg) + ": cusparse error " + std::to_string(s));
}

void checkCusolver(cusolverStatus_t s, const char* msg) {
  if (s != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(msg) + ": cusolver error " + std::to_string(s));
}

}  // namespace

CuSparseSolver::CuSparseSolver() {
  checkCusparse(cusparseCreate(&cusparseH_), "cusparseCreate");
  checkCusolver(cusolverSpCreate(&cusolverH_), "cusolverSpCreate");
  checkCusparse(cusparseCreateMatDescr(&descrAtA_), "cusparseCreateMatDescr");
  cusparseSetMatType(descrAtA_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrAtA_, CUSPARSE_INDEX_BASE_ZERO);
}

CuSparseSolver::~CuSparseSolver() {
  freeAtA();
  if (rhs_) cudaFree(rhs_);
  if (sol_) cudaFree(sol_);
  if (descrAtA_) cusparseDestroyMatDescr(descrAtA_);
  if (cusolverH_) cusolverSpDestroy(cusolverH_);
  if (cusparseH_) cusparseDestroy(cusparseH_);
}

void CuSparseSolver::freeAtA() {
  if (ataRowPtr_) { cudaFree(ataRowPtr_); ataRowPtr_ = nullptr; }
  if (ataColInd_) { cudaFree(ataColInd_); ataColInd_ = nullptr; }
  if (ataVal_) { cudaFree(ataVal_); ataVal_ = nullptr; }
  ataNnz_ = 0;
}

VectorValues CuSparseSolver::solve(const GaussianFactorGraph& gfg,
                                    const Ordering& ordering,
                                    const Scatter& scatter) {
  // Get augmented sparse Jacobian [A | b] as Eigen sparse matrix (CSC format)
  SparseEigen Ab = sparseJacobianEigen(gfg, ordering);
  Ab.makeCompressed();

  const int64_t m = Ab.rows();
  const int64_t n_aug = Ab.cols();
  const int64_t n = n_aug - 1;

  // Copy Jacobian CSC data into managed memory
  // Eigen CSC: outerIndexPtr[n_aug+1], innerIndexPtr[nnz], valuePtr[nnz]
  const int64_t jacNnz = Ab.nonZeros();

  int* d_jacOuterPtr = nullptr;
  int* d_jacInnerIdx = nullptr;
  double* d_jacValues = nullptr;

  checkCuda(cudaMallocManaged(&d_jacOuterPtr, sizeof(int) * (n_aug + 1)), "malloc jacOuter");
  checkCuda(cudaMallocManaged(&d_jacInnerIdx, sizeof(int) * jacNnz), "malloc jacInner");
  checkCuda(cudaMallocManaged(&d_jacValues, sizeof(double) * jacNnz), "malloc jacValues");

  std::memcpy(d_jacOuterPtr, Ab.outerIndexPtr(), sizeof(int) * (n_aug + 1));
  std::memcpy(d_jacInnerIdx, Ab.innerIndexPtr(), sizeof(int) * jacNnz);
  std::memcpy(d_jacValues, Ab.valuePtr(), sizeof(double) * jacNnz);

  // Compute A'b on CPU (sparse column dot — cheap, just column n of Ab transposed)
  // This is A' * b where b is the last column of Ab.
  // Much cheaper than A'A, so keep on CPU.
  Eigen::VectorXd Atb_cpu(n);
  {
    SparseEigen A_part = Ab.leftCols(n);
    Eigen::VectorXd b_part = Eigen::VectorXd(Ab.col(n));
    Atb_cpu = A_part.transpose() * b_part;
  }

  // Allocate RHS/solution buffers
  if (n > allocRhsN_) {
    if (rhs_) cudaFree(rhs_);
    if (sol_) cudaFree(sol_);
    checkCuda(cudaMallocManaged(&rhs_, sizeof(double) * n), "malloc rhs");
    checkCuda(cudaMallocManaged(&sol_, sizeof(double) * n), "malloc sol");
    allocRhsN_ = n;
  }
  std::memcpy(rhs_, Atb_cpu.data(), sizeof(double) * n);

  // Create CSC descriptor for the full augmented Jacobian [A|b] (m x n_aug)
  cusparseSpMatDescr_t matAb = nullptr;
  checkCusparse(cusparseCreateCsc(&matAb, m, n_aug, jacNnz,
      d_jacOuterPtr, d_jacInnerIdx, d_jacValues,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsc Ab");

  // Create CSC descriptor for just A (m x n) — same arrays, just n columns
  // CSC format: first n+1 entries of outerPtr define the A portion
  cusparseSpMatDescr_t matA = nullptr;
  int64_t aNnz = d_jacOuterPtr[n]; // nnz in first n columns
  checkCusparse(cusparseCreateCsc(&matA, m, n, aNnz,
      d_jacOuterPtr, d_jacInnerIdx, d_jacValues,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsc A");

  // We need A^T (n x m) for SpGEMM: C = A^T * A
  // CSC(A) with dims (m,n) interpreted as CSR = A^T (n x m) in CSR format.
  // So create a CSR descriptor with rows=n, cols=m using the same CSC arrays.
  cusparseSpMatDescr_t matAt = nullptr;
  checkCusparse(cusparseCreateCsr(&matAt, n, m, aNnz,
      d_jacOuterPtr, d_jacInnerIdx, d_jacValues,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr At");

  // Create empty CSR descriptor for result C = A^T * A (n x n)
  // Also interpreted as CSR for cuSPARSE, which cusolverSp can consume directly
  cusparseSpMatDescr_t matC = nullptr;
  checkCusparse(cusparseCreateCsr(&matC, n, n, 0,
      nullptr, nullptr, nullptr,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr C");

  // SpGEMM: C = alpha * A^T * A + beta * C
  double alpha = 1.0, beta = 0.0;
  cusparseSpGEMMDescr_t spgemmDesc = nullptr;
  checkCusparse(cusparseSpGEMM_createDescr(&spgemmDesc), "SpGEMM createDescr");

  // Phase 1: work estimation
  size_t bufSize1 = 0;
  checkCusparse(cusparseSpGEMM_workEstimation(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matAt, matA, &beta, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize1, nullptr), "SpGEMM workEst size");

  void* buf1 = nullptr;
  if (bufSize1 > 0) checkCuda(cudaMallocManaged(&buf1, bufSize1), "malloc buf1");

  checkCusparse(cusparseSpGEMM_workEstimation(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matAt, matA, &beta, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize1, buf1), "SpGEMM workEst");

  // Phase 2: compute
  size_t bufSize2 = 0;
  checkCusparse(cusparseSpGEMM_compute(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matAt, matA, &beta, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize2, nullptr), "SpGEMM compute size");

  void* buf2 = nullptr;
  if (bufSize2 > 0) checkCuda(cudaMallocManaged(&buf2, bufSize2), "malloc buf2");

  checkCusparse(cusparseSpGEMM_compute(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matAt, matA, &beta, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize2, buf2), "SpGEMM compute");

  // Phase 3: get result dimensions and copy
  int64_t cRows, cCols, cNnz;
  checkCusparse(cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz), "SpMatGetSize");

  // Allocate result arrays for A'A
  freeAtA();
  ataNnz_ = cNnz;
  checkCuda(cudaMallocManaged(&ataRowPtr_, sizeof(int) * (n + 1)), "malloc ataRowPtr");
  checkCuda(cudaMallocManaged(&ataColInd_, sizeof(int) * cNnz), "malloc ataColInd");
  checkCuda(cudaMallocManaged(&ataVal_, sizeof(double) * cNnz), "malloc ataVal");

  checkCusparse(cusparseCsrSetPointers(matC, ataRowPtr_, ataColInd_, ataVal_),
      "CsrSetPointers");
  checkCusparse(cusparseSpGEMM_copy(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matAt, matA, &beta, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc), "SpGEMM copy");

  cudaDeviceSynchronize();

  // Cleanup SpGEMM temporaries
  cusparseSpGEMM_destroyDescr(spgemmDesc);
  cusparseDestroySpMat(matC);
  cusparseDestroySpMat(matAt);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matAb);
  if (buf2) cudaFree(buf2);
  if (buf1) cudaFree(buf1);
  cudaFree(d_jacValues);
  cudaFree(d_jacInnerIdx);
  cudaFree(d_jacOuterPtr);

  // Cholesky solve: A'A * x = A'b
  // ataRowPtr_/ataColInd_/ataVal_ is CSR format (full matrix, not just triangle)
  // cusolverSpDcsrlsvchol reads lower triangle of CSR input
  cudaDeviceSynchronize();

  int singularity = 0;
  checkCusolver(
      cusolverSpDcsrlsvchol(
          cusolverH_, n, ataNnz_, descrAtA_,
          ataVal_, ataRowPtr_, ataColInd_,
          rhs_, 0.0, 0, sol_, &singularity),
      "cusolverSpDcsrlsvchol");

  cudaDeviceSynchronize();

  if (singularity != -1)
    throw std::runtime_error("CuSparseSolver: matrix is singular at row " +
                             std::to_string(singularity));

  // Construct VectorValues from solution
  Eigen::Map<Eigen::VectorXd> x_vec(sol_, n);
  Eigen::VectorXd solution(x_vec);

  return VectorValues(solution, scatter);
}

VectorValues cuSparseSolve(const GaussianFactorGraph& gfg,
                           const Ordering& ordering,
                           const Scatter& scatter) {
  static thread_local std::unique_ptr<CuSparseSolver> solver;
  if (!solver) solver = std::make_unique<CuSparseSolver>();
  return solver->solve(gfg, ordering, scatter);
}

}  // namespace gtsam
