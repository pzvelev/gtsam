/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <gtsam/linear/CuSparseSolver.h>
#include <gtsam/linear/SparseEigen.h>

#include <cudss.h>
#include <cuda_runtime.h>

// Cast void* members to cuDSS types
#define CUDSS_H static_cast<cudssHandle_t>(cudssH_)
#define CUDSS_CFG static_cast<cudssConfig_t>(cudssConfig_)
#define CUDSS_DATA static_cast<cudssData_t>(cudssData_)

#include <chrono>
#include <cstdio>
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

void checkCudss(cudssStatus_t s, const char* msg) {
  if (s != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(msg) + ": cudss error " + std::to_string(s));
}

}  // namespace

CuSparseSolver::CuSparseSolver() {
  checkCusparse(cusparseCreate(&cusparseH_), "cusparseCreate");
  cudssHandle_t h; checkCudss(cudssCreate(&h), "cudssCreate"); cudssH_ = h;
  cudssConfig_t c; checkCudss(cudssConfigCreate(&c), "cudssConfigCreate"); cudssConfig_ = c;
  cudssData_t d; checkCudss(cudssDataCreate(h, &d), "cudssDataCreate"); cudssData_ = d;
}

CuSparseSolver::~CuSparseSolver() {
  freeAtA();
  if (rhs_) cudaFree(rhs_);
  if (sol_) cudaFree(sol_);
  if (cudssData_) cudssDataDestroy(CUDSS_H, CUDSS_DATA);
  if (cudssConfig_) cudssConfigDestroy(CUDSS_CFG);
  if (cudssH_) cudssDestroy(CUDSS_H);
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
  using Clock = std::chrono::high_resolution_clock;
  auto t0 = Clock::now();

  SparseEigen Ab = sparseJacobianEigen(gfg, ordering);
  Ab.makeCompressed();

  const int64_t m = Ab.rows();
  const int64_t n = Ab.cols() - 1;
  const int64_t jacNnz = Ab.nonZeros();

  auto t1 = Clock::now();

  // Copy Jacobian CSC into managed memory
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

  // Allocate RHS/solution
  if (n > allocRhsN_) {
    if (rhs_) cudaFree(rhs_);
    if (sol_) cudaFree(sol_);
    checkCuda(cudaMallocManaged(&rhs_, sizeof(double) * n), "malloc rhs");
    checkCuda(cudaMallocManaged(&sol_, sizeof(double) * n), "malloc sol");
    allocRhsN_ = n;
  }

  auto t2 = Clock::now();

  // === A'b via cusparseSpMV on GPU ===
  double* d_bVec = nullptr;
  checkCuda(cudaMallocManaged(&d_bVec, sizeof(double) * m), "malloc bVec");
  cudaMemset(d_bVec, 0, sizeof(double) * m);

  int bStart = d_cscColPtr[n];
  int bEnd = d_cscColPtr[n + 1];
  for (int j = bStart; j < bEnd; j++)
    d_bVec[d_cscRowIdx[j]] = d_cscVal[j];

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

  auto t3 = Clock::now();

  // === A'A via cuSPARSE SpGEMM ===
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

  if (convBuf) cudaFree(convBuf);

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

  auto t4 = Clock::now();

  // === Cholesky solve via cuDSS ===
  cudssDataDestroy(CUDSS_H, CUDSS_DATA);
  { cudssData_t d; cudssDataCreate(CUDSS_H, &d); cudssData_ = d; }

  cudssMatrix_t cudssAtA = nullptr;
  checkCudss(cudssMatrixCreateCsr(&cudssAtA, n, n, ataNnz_,
      ataRowPtr_, nullptr, ataColInd_, ataVal_,
      CUDA_R_32I, CUDA_R_64F,
      CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
      "cudssMatrixCreateCsr");

  cudssMatrix_t cudssRhs = nullptr;
  checkCudss(cudssMatrixCreateDn(&cudssRhs, n, 1, n, rhs_,
      CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn rhs");

  cudssMatrix_t cudssSol = nullptr;
  checkCudss(cudssMatrixCreateDn(&cudssSol, n, 1, n, sol_,
      CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn sol");

  checkCudss(cudssExecute(CUDSS_H, CUDSS_PHASE_ANALYSIS, CUDSS_CFG, CUDSS_DATA,
      cudssAtA, cudssSol, cudssRhs), "cudss analysis");

  checkCudss(cudssExecute(CUDSS_H, CUDSS_PHASE_FACTORIZATION, CUDSS_CFG, CUDSS_DATA,
      cudssAtA, cudssSol, cudssRhs), "cudss factorization");

  checkCudss(cudssExecute(CUDSS_H, CUDSS_PHASE_SOLVE, CUDSS_CFG, CUDSS_DATA,
      cudssAtA, cudssSol, cudssRhs), "cudss solve");

  cudaDeviceSynchronize();

  cudssMatrixDestroy(cudssSol);
  cudssMatrixDestroy(cudssRhs);
  cudssMatrixDestroy(cudssAtA);

  auto t5 = Clock::now();

  Eigen::Map<Eigen::VectorXd> x_vec(sol_, n);
  Eigen::VectorXd solution(x_vec);

  auto t6 = Clock::now();

  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  static int callCount = 0;
  if (callCount++ % 5 == 0) {
    std::fprintf(stderr,
        "[CuDSS] jacobian=%.2f memcpy=%.2f SpMV=%.2f SpGEMM=%.2f cholesky=%.2f result=%.2f TOTAL=%.2f ms\n",
        ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t5, t6), ms(t0, t6));
  }

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
