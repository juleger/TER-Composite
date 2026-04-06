#include "test_shear.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// TEST CISAILLEMENT PUR
void runShearTest(const string& meshFile, const Config& config) {
    runShearTest(vector<string>{meshFile}, vector<double>{1.0}, config);
}

// TEST CISAILLEMENT GAUSSIEN - CONVERGENCE MAILLAGE
// Chargement tangentiel non uniforme sur le bord supérieur
// Pas de solution analytique fermée 2D: la convergence est suivie via deltaW_rel.
void runShearTest(const vector<string>& meshFiles, const vector<double>& meshLc, const Config& config) {
    cout << "---------- CISAILLEMENT GAUSSIEN - CONVERGENCE MAILLAGE ----------" << endl;

    Material material(config.E, config.nu, config.rho);
    vector<ConvergenceResult> results = runConvergence(
        meshFiles, meshLc,
        [&](size_t i, const string& mfile, double h, ConvergenceResult& out) {
            Mesh mesh;
            loadAndInitMesh(mfile, material, mesh);

            Solver solver(mesh, config.solverTolerance, config.solverMaxIter);
            applySolverConfig(solver, config);
            solver.assemble();

            for (int id : mesh.bottomNodes) {
                solver.setDirichletBC(id, 0, 0.0);
                solver.setDirichletBC(id, 1, 0.0);
            }

            const double L     = mesh.width();
            const double H     = mesh.height();
            const double xMid  = 0.5 * (mesh.xMin + mesh.xMax);
            const double sigma = L / 4.0;
            const double F_tot = config.forceValue;

            const double sq2 = std::sqrt(2.0);
            const double gaussIntegral = sigma * std::sqrt(2.0 * M_PI) *
                (std::erf((mesh.xMax - xMid) / (sigma * sq2))
               - std::erf((mesh.xMin - xMid) / (sigma * sq2))) / 2.0;
            const double A = F_tot / max(gaussIntegral, 1e-30);

            applyDistributedForce(solver, mesh, mesh.topNodes,
                [A, xMid, sigma](double x) {
                    return A * std::exp(-((x - xMid) * (x - xMid)) / (2.0 * sigma * sigma));
                }, 0);

            solver.applyBC();
            solver.solveConjugateGradient();
            solver.computeStrainStress();
            solver.saveVTK(convergenceVtkPath(config, h));

            const Eigen::VectorXd U = solver.getU();
            double uxTop = 0.0;
            for (int id : mesh.topNodes) uxTop += U(2*(id-1));
            uxTop /= max<double>(mesh.topNodes.size(), 1.0);
            const double gammaApp = uxTop / max(H, 1e-30);
            const double tauMean  = F_tot / max(L, 1e-30);
            const double Gapp     = abs(tauMean) / max(abs(gammaApp), 1e-30);

            const double W_int = solver.computeInternalEnergy();
            const double W_ext = solver.computeExternalWork();
            const double deltaWrel = abs(W_int - W_ext) / max(abs(W_int), 1e-30);

            cout << "deltaW_rel=" << deltaWrel << ", G_app=" << Gapp
                 << ", gamma_app=" << gammaApp << endl;

            out = {h, mesh.nbElements(), -1.0, deltaWrel, -1.0, -1.0, -1.0};
            return true;
        });

    ReportOptions reportOpt;
    reportOpt.showL2     = false;
    reportOpt.showDeltaW = true;
    reportOpt.showErrG   = false;
    printConvergenceTable(results, "Cisaillement Gaussien", reportOpt,
        [](const ConvergenceResult& r){ return r.deltaWrel; });
    exportConvergenceCSV(results, "results/convergence_shear.csv", reportOpt,
        [](const ConvergenceResult& r){ return r.deltaWrel; });
}
