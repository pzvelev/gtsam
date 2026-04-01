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

#include <algorithm>
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

// Permute a symmetric CSR matrix: P * A * P^T
// and permute RHS: P * b
// Both on CPU since AMD ordering is already on CPU and this is cheap
void permuteSymmetricCSR(
    const int* rowPtr, const int* colInd, const double* val,
    int n, int64_t nnz,
    const int* perm, const int* permInv,
    int* outRowPtr, int* outColInd, double* outVal,
    const double* rhs, double* outRhs) {

  // Count nnz per permuted row
  std::vector<int> rowCount(n, 0);
  for (int i = 0; i < n; i++) {
    int pi = perm[i];  // new row index
    rowCount[pi] = rowPtr[i + 1] - rowPtr[i];
  }
  outRowPtr[0] = 0;
  for (int i = 0; i < n; i++)
    outRowPtr[i + 1] = outRowPtr[i] + rowCount[i];

  // Fill permuted entries
  std::vector<int> offset(n, 0);
  for (int oldRow = 0; oldRow < n; oldRow++) {
    int newRow = perm[oldRow];
    int dest = outRowPtr[newRow] + offset[newRow];
    for (int j = rowPtr[oldRow]; j < rowPtr[oldRow + 1]; j++) {
      int oldCol = colInd[j];
      int newCol = perm[oldCol];
      outColInd[dest] = newCol;
      outVal[dest] = val[j];
      dest++;
    }
    offset[newRow] = dest - outRowPtr[newRow];
  }

  // Sort each row by column index (cusolverSp requires sorted columns)
  for (int i = 0; i < n; i++) {
    int start = outRowPtr[i];
    int end = outRowPtr[i + 1];
    // Simple insertion sort (rows are typically short in SLAM)
    for (int j = start + 1; j < end; j++) {
      int keyCol = outColInd[j];
      double keyVal = outVal[j];
      int k = j - 1;
      while (k >= start && outColInd[k] > keyCol) {
        outColInd[k + 1] = outColInd[k];
        outVal[k + 1] = outVal[k];
        k--;
      }
      outColInd[k + 1] = keyCol;
      outVal[k + 1] = keyVal;
    }
  }

  // Permute RHS
  for (int i = 0; i < n; i++)
    outRhs[perm[i]] = rhs[i];
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
  if (d_perm_) cudaFree(d_perm_);
  if (d_permInv_) cudaFree(d_permInv_);
  if (permAtaRowPtr_) cudaFree(permAtaRowPtr_);
  if (permAtaColInd_) cudaFree(permAtaColInd_);
  if (permAtaVal_) cudaFree(permAtaVal_);
  if (permRhs_) cudaFree(permRhs_);
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

  cudaDeviceSynchronize();
  int64_t aNnz = d_cscColPtr[n];

  // Allocate RHS/solution buffers
  if (n > allocRhsN_) {
    if (rhs_) cudaFree(rhs_);
    if (sol_) cudaFree(sol_);
    if (permRhs_) cudaFree(permRhs_);
    checkCuda(cudaMallocManaged(&rhs_, sizeof(double) * n), "malloc rhs");
    checkCuda(cudaMallocManaged(&sol_, sizeof(double) * n), "malloc sol");
    checkCuda(cudaMallocManaged(&permRhs_, sizeof(double) * n), "malloc permRhs");
    allocRhsN_ = n;
  }

  // === A'b via cusparseSpMV on GPU ===
  // CSC(A) reinterpreted as CSR(A^T) [n×m], multiply by b vector [m×1] → A'b [n×1]
  // b is the last column of Ab in CSC — extract the values for rows in column n
  double* d_bVec = nullptr;
  checkCuda(cudaMallocManaged(&d_bVec, sizeof(double) * m), "malloc bVec");
  cudaMemset(d_bVec, 0, sizeof(double) * m);

  // Extract b column from CSC: entries in column n are at indices [colPtr[n], colPtr[n+1])
  int bStart = d_cscColPtr[n];
  int bEnd = d_cscColPtr[n + 1];
  for (int j = bStart; j < bEnd; j++)
    d_bVec[d_cscRowIdx[j]] = d_cscVal[j];

  // Create CSR descriptor for A^T (reinterpret CSC(A) as CSR(A^T))
  cusparseSpMatDescr_t matAt_mv = nullptr;
  checkCusparse(cusparseCreateCsr(&matAt_mv, n, m, aNnz,
      d_cscColPtr, d_cscRowIdx, d_cscVal,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr At_mv");

  cusparseDnVecDescr_t vecB = nullptr, vecAtb = nullptr;
  checkCusparse(cusparseCreateDnVec(&vecB, m, d_bVec, CUDA_R_64F), "createDnVec b");
  checkCusparse(cusparseCreateDnVec(&vecAtb, n, rhs_, CUDA_R_64F), "createDnVec Atb");

  double one = 1.0, zero = 0.0;

  size_t mvBufSize = 0;
  checkCusparse(cusparseSpMV_bufferSize(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAt_mv, vecB, &zero, vecAtb,
      CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &mvBufSize), "SpMV bufsize");

  void* mvBuf = nullptr;
  if (mvBufSize > 0) checkCuda(cudaMallocManaged(&mvBuf, mvBufSize), "malloc mvBuf");

  checkCusparse(cusparseSpMV(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAt_mv, vecB, &zero, vecAtb,
      CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, mvBuf), "SpMV Atb");

  cusparseDestroyDnVec(vecAtb);
  cusparseDestroyDnVec(vecB);
  cusparseDestroySpMat(matAt_mv);
  if (mvBuf) cudaFree(mvBuf);
  cudaFree(d_bVec);

  // === A'A via cuSPARSE SpGEMM on GPU ===
  // Convert A from CSC to CSR on GPU
  int* d_csrRowPtr = nullptr;
  int* d_csrColInd = nullptr;
  double* d_csrVal = nullptr;

  checkCuda(cudaMallocManaged(&d_csrRowPtr, sizeof(int) * (m + 1)), "malloc csrRowPtr");
  checkCuda(cudaMallocManaged(&d_csrColInd, sizeof(int) * aNnz), "malloc csrColInd");
  checkCuda(cudaMallocManaged(&d_csrVal, sizeof(double) * aNnz), "malloc csrVal");

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

  // SpGEMM: C = A^T * A (both operands in CSR)
  cusparseSpMatDescr_t matAt = nullptr;
  checkCusparse(cusparseCreateCsr(&matAt, n, m, aNnz,
      d_cscColPtr, d_cscRowIdx, d_cscVal,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr At");

  cusparseSpMatDescr_t matA = nullptr;
  checkCusparse(cusparseCreateCsr(&matA, m, n, aNnz,
      d_csrRowPtr, d_csrColInd, d_csrVal,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr A");

  cusparseSpMatDescr_t matC = nullptr;
  checkCusparse(cusparseCreateCsr(&matC, n, n, 0,
      nullptr, nullptr, nullptr,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "createCsr C");

  cusparseSpGEMMDescr_t spgemmDesc = nullptr;
  checkCusparse(cusparseSpGEMM_createDescr(&spgemmDesc), "SpGEMM createDescr");

  size_t bufSize1 = 0;
  checkCusparse(cusparseSpGEMM_workEstimation(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matAt, matA, &zero, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize1, nullptr), "SpGEMM workEst size");

  void* buf1 = nullptr;
  if (bufSize1 > 0) checkCuda(cudaMallocManaged(&buf1, bufSize1), "malloc buf1");

  checkCusparse(cusparseSpGEMM_workEstimation(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matAt, matA, &zero, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize1, buf1), "SpGEMM workEst");

  size_t bufSize2 = 0;
  checkCusparse(cusparseSpGEMM_compute(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matAt, matA, &zero, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize2, nullptr), "SpGEMM compute size");

  void* buf2 = nullptr;
  if (bufSize2 > 0) checkCuda(cudaMallocManaged(&buf2, bufSize2), "malloc buf2");

  checkCusparse(cusparseSpGEMM_compute(cusparseH_,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matAt, matA, &zero, matC,
      CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufSize2, buf2), "SpGEMM compute");

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
      &one, matAt, matA, &zero, matC,
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

  // === AMD ordering for fill reduction ===
  // Build Eigen sparse matrix from CSR A'A (A'A is symmetric so CSR=CSC transposed)
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(ataNnz_);
  for (int i = 0; i < n; i++)
    for (int j = ataRowPtr_[i]; j < ataRowPtr_[i + 1]; j++)
      trips.emplace_back(i, ataColInd_[j], ataVal_[j]);

  SparseEigen ataEigen(n, n);
  ataEigen.setFromTriplets(trips.begin(), trips.end());

  Eigen::AMDOrdering<int> amd;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permMat;
  amd(ataEigen.selfadjointView<Eigen::Upper>(), permMat);

  if (allocPermN_ < n) {
    if (d_perm_) cudaFree(d_perm_);
    if (d_permInv_) cudaFree(d_permInv_);
    checkCuda(cudaMallocManaged(&d_perm_, sizeof(int) * n), "malloc perm");
    checkCuda(cudaMallocManaged(&d_permInv_, sizeof(int) * n), "malloc permInv");
    allocPermN_ = n;
  }
  std::memcpy(d_perm_, permMat.indices().data(), sizeof(int) * n);
  auto permInvMat = permMat.inverse();
  std::memcpy(d_permInv_, permInvMat.indices().data(), sizeof(int) * n);

  // === Apply permutation: P * A'A * P^T and P * A'b ===
  if (permAtaNnz_ < ataNnz_ || permAtaRowPtr_ == nullptr) {
    if (permAtaRowPtr_) cudaFree(permAtaRowPtr_);
    if (permAtaColInd_) cudaFree(permAtaColInd_);
    if (permAtaVal_) cudaFree(permAtaVal_);
    checkCuda(cudaMallocManaged(&permAtaRowPtr_, sizeof(int) * (n + 1)), "malloc permAtaRowPtr");
    checkCuda(cudaMallocManaged(&permAtaColInd_, sizeof(int) * ataNnz_), "malloc permAtaColInd");
    checkCuda(cudaMallocManaged(&permAtaVal_, sizeof(double) * ataNnz_), "malloc permAtaVal");
    permAtaNnz_ = ataNnz_;
  }

  permuteSymmetricCSR(ataRowPtr_, ataColInd_, ataVal_,
      n, ataNnz_, d_perm_, d_permInv_,
      permAtaRowPtr_, permAtaColInd_, permAtaVal_,
      rhs_, permRhs_);

  cudaDeviceSynchronize();

  // === Cholesky solve on permuted system ===
  int singularity = 0;
  checkCusolver(
      cusolverSpDcsrlsvchol(
          cusolverH_, n, ataNnz_, descrAtA_,
          permAtaVal_, permAtaRowPtr_, permAtaColInd_,
          permRhs_, 0.0, 0, sol_, &singularity),
      "cusolverSpDcsrlsvchol");

  cudaDeviceSynchronize();

  if (singularity != -1)
    throw std::runtime_error("CuSparseSolver: matrix is singular at row " +
                             std::to_string(singularity));

  // Inverse-permute solution: x[permInv[i]] = sol[i]
  Eigen::VectorXd solution(n);
  for (int i = 0; i < n; i++)
    solution[d_permInv_[i]] = sol_[i];

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
