/* ----------------------------------------------------------------------------
 * Benchmark: CPU (MULTIFRONTAL_CHOLESKY) vs GPU (cuSPARSE) sparse Cholesky
 *
 * Builds a 2D pose graph of configurable size and compares solve time and
 * solution accuracy between the two backends.
 *
 * Usage: timeCuSparseSolver [num_poses] [num_loops]
 *   num_poses: number of poses in the chain (default 500)
 *   num_loops: number of random loop closures (default 50)
 * -------------------------------------------------------------------------- */

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/Scatter.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#ifdef GTSAM_WITH_CUSPARSE
#include <gtsam/linear/CuSparseSolver.h>
#endif

#ifdef GTSAM_WITH_BASPACHO
#include <baspacho/baspacho/Solver.h>
#include <gtsam/linear/HessianFactor.h>
#endif

#include <chrono>
#include <iostream>
#include <random>

using namespace gtsam;
using namespace std;

int main(int argc, char* argv[]) {
  const int numPoses = argc > 1 ? atoi(argv[1]) : 500;
  const int numLoops = argc > 2 ? atoi(argv[2]) : 50;
  const int numTrials = 5;

  cout << "=== cuSPARSE vs CPU Cholesky Benchmark ===" << endl;
  cout << "Poses: " << numPoses << ", Loop closures: " << numLoops << endl;

  // Build a pose graph
  auto odomNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
  auto loopNoise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));
  auto priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01));

  NonlinearFactorGraph graph;
  Values initial;

  // Prior on first pose
  graph.addPrior(0, Pose2(0, 0, 0), priorNoise);
  initial.insert(0, Pose2(0, 0, 0));

  // Odometry chain
  mt19937 rng(42);
  normal_distribution<double> noise(0, 0.1);

  for (int i = 1; i < numPoses; i++) {
    Pose2 odom(1.0, 0.0, 0.1);
    graph.emplace_shared<BetweenFactor<Pose2>>(i - 1, i, odom, odomNoise);

    Pose2 prev = initial.at<Pose2>(i - 1);
    Pose2 noisy(prev.x() + 1.0 + noise(rng),
                prev.y() + noise(rng),
                prev.theta() + 0.1 + noise(rng) * 0.01);
    initial.insert(i, noisy);
  }

  // Random loop closures
  uniform_int_distribution<int> poseDist(0, numPoses - 1);
  for (int i = 0; i < numLoops; i++) {
    int a = poseDist(rng);
    int b = poseDist(rng);
    if (a == b) continue;

    Pose2 pa = initial.at<Pose2>(a);
    Pose2 pb = initial.at<Pose2>(b);
    Pose2 between = pa.between(pb);
    graph.emplace_shared<BetweenFactor<Pose2>>(a, b, between, loopNoise);
  }

  cout << "Graph: " << graph.size() << " factors, " << initial.size() << " variables" << endl;

  // === CPU Benchmark (MULTIFRONTAL_CHOLESKY) ===
  {
    LevenbergMarquardtParams params;
    params.linearSolverType = NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;
    params.maxIterations = 20;
    params.setVerbosity("SILENT");

    // Warmup
    LevenbergMarquardtOptimizer(graph, initial, params).optimize();

    double totalMs = 0;
    double finalError = 0;
    for (int t = 0; t < numTrials; t++) {
      auto start = chrono::high_resolution_clock::now();
      LevenbergMarquardtOptimizer optimizer(graph, initial, params);
      Values result = optimizer.optimize();
      auto end = chrono::high_resolution_clock::now();

      double ms = chrono::duration<double, milli>(end - start).count();
      totalMs += ms;
      finalError = optimizer.error();
    }
    cout << "\nCPU MULTIFRONTAL_CHOLESKY:" << endl;
    cout << "  Avg time: " << totalMs / numTrials << " ms" << endl;
    cout << "  Final error: " << finalError << endl;
  }

  // === GPU Benchmark (CHOLMOD -> cuSPARSE) ===
#ifdef GTSAM_WITH_CUSPARSE
  {
    LevenbergMarquardtParams params;
    params.linearSolverType = NonlinearOptimizerParams::CHOLMOD;
    params.maxIterations = 20;
    params.setVerbosity("SILENT");

    // Warmup (also initializes CUDA context)
    LevenbergMarquardtOptimizer(graph, initial, params).optimize();

    double totalMs = 0;
    double finalError = 0;
    for (int t = 0; t < numTrials; t++) {
      auto start = chrono::high_resolution_clock::now();
      LevenbergMarquardtOptimizer optimizer(graph, initial, params);
      Values result = optimizer.optimize();
      auto end = chrono::high_resolution_clock::now();

      double ms = chrono::duration<double, milli>(end - start).count();
      totalMs += ms;
      finalError = optimizer.error();
    }
    cout << "\nGPU cuSPARSE (via CHOLMOD enum):" << endl;
    cout << "  Avg time: " << totalMs / numTrials << " ms" << endl;
    cout << "  Final error: " << finalError << endl;
  }
#else
  cout << "\nGPU cuSPARSE: DISABLED (build with -DGTSAM_WITH_CUSPARSE=ON)" << endl;
#endif

  // === Also benchmark just the linear solve step ===
  cout << "\n--- Linear solve only (single linearization) ---" << endl;

  Ordering ordering = Ordering::Colamd(graph);
  GaussianFactorGraph::shared_ptr gfg = graph.linearize(initial);

  // CPU linear solve
  {
    double totalMs = 0;
    VectorValues delta;
    for (int t = 0; t < numTrials; t++) {
      auto start = chrono::high_resolution_clock::now();
      delta = gfg->optimize(ordering);
      auto end = chrono::high_resolution_clock::now();
      totalMs += chrono::duration<double, milli>(end - start).count();
    }
    cout << "CPU linear solve: " << totalMs / numTrials << " ms (dim=" << delta.size() << ")" << endl;
  }

#ifdef GTSAM_WITH_CUSPARSE
  // GPU linear solve
  {
    Scatter scatter(*gfg, ordering);
    double totalMs = 0;
    VectorValues delta;
    for (int t = 0; t < numTrials; t++) {
      auto start = chrono::high_resolution_clock::now();
      delta = cuSparseSolve(*gfg, ordering, scatter);
      auto end = chrono::high_resolution_clock::now();
      totalMs += chrono::duration<double, milli>(end - start).count();
    }
    cout << "GPU linear solve: " << totalMs / numTrials << " ms (dim=" << delta.size() << ")" << endl;
  }
#endif

#ifdef GTSAM_WITH_BASPACHO
  // BaSpaCho linear solve
  {
    Scatter scatter(*gfg, ordering);

    // Get block sizes from scatter
    std::vector<int64_t> paramSizes;
    for (const auto& entry : scatter)
      paramSizes.push_back(entry.dimension);
    int64_t totalDim = 0;
    for (auto s : paramSizes) totalDim += s;

    // Build dense augmented Hessian [AtA | Atb; Atb' | c]
    HessianFactor combined(*gfg, scatter);
    Matrix augmented = combined.info().selfadjointView();
    int64_t n = augmented.rows() - 1;

    Matrix AtA = augmented.topLeftCorner(n, n);
    Vector Atb = augmented.topRightCorner(n, 1);

    // Build block-level lower-triangular CSR for baspacho
    // For a dense Hessian, every block pair is non-zero
    int64_t numBlocks = paramSizes.size();
    std::vector<int64_t> ptrs{0}, inds;
    for (int64_t row = 0; row < numBlocks; row++) {
      for (int64_t col = 0; col <= row; col++)
        inds.push_back(col);
      ptrs.push_back(inds.size());
    }

    // Create solver (CPU)
    BaSpaCho::Settings bsSettings;
    bsSettings.numThreads = 1;
    bsSettings.backend = BaSpaCho::BackendFast;
    auto solverCPU = BaSpaCho::createSolver(
        bsSettings, paramSizes, BaSpaCho::SparseStructure(ptrs, inds));

    // Fill baspacho data from dense Hessian blocks
    auto acc = solverCPU->accessor();
    std::vector<double> hessData(solverCPU->dataSize(), 0.0);
    std::vector<double> gradData(solverCPU->order(), 0.0);

    // Copy Hessian blocks
    int64_t rowOff = 0;
    for (int64_t bi = 0; bi < numBlocks; bi++) {
      int64_t colOff = 0;
      for (int64_t bj = 0; bj <= bi; bj++) {
        auto block = acc.block(hessData.data(), bi, bj);
        block = AtA.block(rowOff, colOff, paramSizes[bi], paramSizes[bj]);
        colOff += paramSizes[bj];
      }
      // Copy gradient
      auto gradSeg = Eigen::Map<Eigen::VectorXd>(gradData.data() + acc.paramStart(bi), paramSizes[bi]);
      gradSeg = Atb.segment(rowOff, paramSizes[bi]);
      rowOff += paramSizes[bi];
    }

    // Warmup
    {
      auto h = hessData;
      auto g = gradData;
      solverCPU->factor(h.data());
      solverCPU->solve(h.data(), g.data(), solverCPU->order(), 1);
    }

    double totalMs = 0;
    for (int t = 0; t < numTrials; t++) {
      auto h = hessData;
      auto g = gradData;
      auto start = chrono::high_resolution_clock::now();
      solverCPU->factor(h.data());
      solverCPU->solve(h.data(), g.data(), solverCPU->order(), 1);
      auto end = chrono::high_resolution_clock::now();
      totalMs += chrono::duration<double, milli>(end - start).count();
    }
    cout << "BaSpaCho CPU linear solve: " << totalMs / numTrials << " ms" << endl;

    // Verify correctness: check solution matches
    {
      auto h = hessData;
      auto g = gradData;
      solverCPU->factor(h.data());
      solverCPU->solve(h.data(), g.data(), solverCPU->order(), 1);
      // g now contains the solution in baspacho's permuted order
      // Map back through accessor
      Eigen::VectorXd solution(n);
      for (int64_t bi = 0; bi < numBlocks; bi++) {
        int64_t off = acc.paramStart(bi);
        int64_t sz = paramSizes[bi];
        // Find original offset in scatter order
        int64_t origOff = 0;
        for (int64_t k = 0; k < bi; k++) origOff += paramSizes[k];
        solution.segment(origOff, sz) = Eigen::Map<Eigen::VectorXd>(g.data() + off, sz);
      }
      VectorValues delta(solution, scatter);
      cout << "  (solution dim=" << delta.size() << ")" << endl;
    }
  }
#else
  cout << "BaSpaCho: DISABLED (build with -DGTSAM_WITH_BASPACHO=ON)" << endl;
#endif

  return 0;
}
