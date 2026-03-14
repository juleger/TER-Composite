#include "test_shear.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// TEST CISAILLEMENT PUR
void runShearTest(const string& meshFile, const Config& config) {
    runShearTest(vector<string>{meshFile}, vector<double>{1.0}, config);
}

// TEST CISAILLEMENT - CONVERGENCE MAILLAGE
void runShearTest(const vector<string>& meshFiles,
                  const vector<double>& meshLc,
                  const Config& config) {
    cout << "-------------- CISAILLEMENT - CONVERGENCE MAILLAGE --------------" << endl;

    Material material(config.E, config.nu, config.rho);
    vector<ConvergenceResult> results = runConvergence(
        meshFiles, meshLc,
        [&](size_t i, const string& mfile, double h, ConvergenceResult& out) {
            Mesh mesh;
            loadAndInitMesh(mfile, material, mesh);

            Solver solver(mesh, config.solverTolerance, config.solverMaxIter);
            applySolverConfig(solver, config);
            solver.assemble();

            const double gammaTarget = 1.0e-3;
            applyShearDirichletBC(solver, mesh, gammaTarget);

            solver.applyBC();
            solver.solveConjugateGradient();
            solver.computeStrainStress();
            solver.saveVTK(convergenceVtkPath(config, h));

            ExactFn exact = [gammaTarget](double, double y) -> Eigen::Vector2d {
                return {gammaTarget * y, 0.0};
            };
            solver.computeL2Error(exact);

            const Eigen::VectorXd U = solver.getU();
            const double V = max(mesh.width() * mesh.height(), 1e-30);
            const double Wint = solver.computeInternalEnergy();
            const double Geff = 2.0 * Wint / max(gammaTarget * gammaTarget * V, 1e-30);
            const double Gtheo = config.E / (2.0 * (1.0 + config.nu));
            const double relErrG = abs(Geff - Gtheo) / max(abs(Gtheo), 1e-30);

            const double l2Rel = solver.errL2_rel;

            cout << "Wint=" << Wint << ", Geff=" << Geff << ", Gtheo=" << Gtheo << ", errG_rel=" << relErrG << ", L2_rel=" << l2Rel << endl;
            out = {h, mesh.nbElements(), l2Rel, -1.0, -1.0, -1.0, relErrG};
            return true;
        });

    ReportOptions reportOpt;
    reportOpt.showL2 = true;
    reportOpt.showDeltaW = false;
    reportOpt.showErrG = true;
    printConvergenceTable(results, "Shear", reportOpt, [](const ConvergenceResult& r){ return r.L2rel; });
    exportConvergenceCSV(results, "results/convergence_shear.csv", reportOpt, [](const ConvergenceResult& r){ return r.L2rel; });
}
