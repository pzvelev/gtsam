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
  // Get augmented sparse Jacobian [A | b] as Eigen CSC sparse matrix
  SparseEigen Ab = sparseJacobianEigen(gfg, ordering);
  Ab.makeCompressed();

  const int64_t m = Ab.rows();
  const int64_t n = Ab.cols() - 1;
  const int64_t jacNnz = Ab.nonZeros();

  // Copy Jacobian CSC arrays into managed memory
  int* d_cscColPtr = nullptr;
  int* d_cscRowIdx = nullptr;
  double* d_cscVal = nullptr;

  checkCuda(cudaMallocManaged(&d_cscColPtr, sizeof(int) * (Ab.cols() + 1)), "malloc cscColPtr");
  checkCuda(cudaMallocManaged(&d_cscRowIdx, sizeof(int) * jacNnz), "malloc cscRowIdx");
  checkCuda(cudaMallocManaged(&d_cscVal, sizeof(double) * jacNnz), "malloc cscVal");

  std::memcpy(d_cscColPtr, Ab.outerIndexPtr(), sizeof(int) * (Ab.cols() + 1));
  std::memcpy(d_cscRowIdx, Ab.innerIndexPtr(), sizeof(int) * jacNnz);
  std::memcpy(d_cscVal, Ab.valuePtr(), sizeof(double) * jacNnz);

  // A is the first n columns of Ab. In CSC:
  //   colPtr = d_cscColPtr[0..n]  (n+1 entries)
  //   nnz(A) = d_cscColPtr[n]
  cudaDeviceSynchronize();
  int64_t aNnz = d_cscColPtr[n];

  // Compute A'b on CPU (cheap — sparse column dot product)
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

  // Convert A from CSC to CSR on GPU using cusparseCsr2cscEx2
  // (CSC→CSR is the same as CSR→CSC with transposed dimensions)
  int* d_csrRowPtr = nullptr;
  int* d_csrColInd = nullptr;
  double* d_csrVal = nullptr;

  checkCuda(cudaMallocManaged(&d_csrRowPtr, sizeof(int) * (m + 1)), "malloc csrRowPtr");
  checkCuda(cudaMallocManaged(&d_csrColInd, sizeof(int) * aNnz), "malloc csrColInd");
  checkCuda(cudaMallocManaged(&d_csrVal, sizeof(double) * aNnz), "malloc csrVal");

  // Convert CSC(A) [m×n] → CSR(A) [m×n] on GPU.
  // cusparseCsr2cscEx2 converts CSR→CSC. We feed it CSC(A) as if it were
  // CSR(A^T) [n×m], and it outputs CSC(A^T) [n×m] = CSR(A) [m×n].
  size_t convBufSize = 0;
  checkCusparse(cusparseCsr2cscEx2_bufferSize(cusparseH_,
      n, m, aNnz,
      d_cscVal, d_cscColPtr, d_cscRowIdx,
      d_csrVal, d_csrRowPtr, d_csrColInd,
      CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
      &convBufSize), "csr2csc bufsize");

  void* convBuf = nullptr;
  if (convBufSize > 0) checkCuda(cudaMallocManaged(&convBuf, convBufSize), "malloc convBuf");

  checkCusparse(cusparseCsr2cscEx2(cusparseH_,
      n, m, aNnz,
      d_cscVal, d_cscColPtr, d_cscRowIdx,
      d_csrVal, d_csrRowPtr, d_csrColInd,
      CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
      convBuf), "csc2csr convert");

  if (convBuf) { cudaFree(convBuf); convBuf = nullptr; }

  // Now we have:
  //   CSR(A^T) [n×m]: reinterpret CSC(A) arrays as CSR → colPtr=rowPtr, rowIdx=colInd
  //   CSR(A)   [m×n]: from the conversion above
  // SpGEMM: C = A^T * A (both in CSR)

  cusparseSpMatDescr_t matAt = nullptr;
  checkCusparse(cusparseCreateCsr(&matAt, n, m, aNnz,
      d_cscColPtr, d_cscRowIdx, d_cscVal,  // CSC(A) reinterpreted as CSR(A^T)
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr At");

  cusparseSpMatDescr_t matA = nullptr;
  checkCusparse(cusparseCreateCsr(&matA, m, n, aNnz,
      d_csrRowPtr, d_csrColInd, d_csrVal,  // Converted CSR(A)
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr A");

  cusparseSpMatDescr_t matC = nullptr;
  checkCusparse(cusparseCreateCsr(&matC, n, n, 0,
      nullptr, nullptr, nullptr,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr C");

  // SpGEMM: C = A^T * A
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

  // Phase 3: extract result
  int64_t cRows, cCols, cNnz;
  checkCusparse(cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz), "SpMatGetSize");

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
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matAt);
  if (buf2) cudaFree(buf2);
  if (buf1) cudaFree(buf1);
  cudaFree(d_csrVal);
  cudaFree(d_csrColInd);
  cudaFree(d_csrRowPtr);
  cudaFree(d_cscVal);
  cudaFree(d_cscRowIdx);
  cudaFree(d_cscColPtr);

  // Cholesky solve: A'A * x = A'b
  // ataRowPtr_/ataColInd_/ataVal_ is full CSR (A'A is SPD)
  // cusolverSpDcsrlsvchol reads the lower triangle
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
