#include "test_flexion.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// TEST FLEXION - CONVERGENCE MAILLAGE
void runFlexionTest(const vector<string>& meshFiles,
                    const vector<double>& meshLc,
                    const Config& config) {
    cout << "---------------- FLEXION - CONVERGENCE MAILLAGE ----------------" << endl;

    Material material(config.E, config.nu, config.rho);
    vector<ConvergenceResult> results = runConvergence(
        meshFiles, meshLc,
        [&](size_t i, const string& mfile, double h, ConvergenceResult& out) {
            Mesh mesh;
            loadAndInitMesh(mfile, material, mesh);

            Solver solver(mesh, config.solverTolerance, config.solverMaxIter);
            applySolverConfig(solver, config);
            solver.assemble();

            for (int id : mesh.leftNodes) {
                solver.setDirichletBC(id, 0, 0.0);
                solver.setDirichletBC(id, 1, 0.0);
            }

            const double H    = mesh.height();
            const double L    = mesh.width();
            const double I    = H * H * H / 12.0;
            const double yMid = 0.5 * (mesh.yMin + mesh.yMax);
            const double F    = config.forceValue;

            applyDistributedForce(solver, mesh, mesh.rightNodes, [F, H, I, yMid](double y) {
                    const double yn = y - yMid;
                    return F * (H*H/4.0 - yn*yn) / (2.0 * I);
                }, 1);

            solver.applyBC();
            solver.solveConjugateGradient();
            solver.computeStrainStress();
            solver.saveVTK(convergenceVtkPath(config, h));

            const double EI = config.E * I;
            ExactFn exact = [F, L, EI, yMid](double x, double y) -> Eigen::Vector2d {
                const double yn = y - yMid;
                return { -F * x * yn * (2.0*L - x) / (2.0*EI),
                          F * x*x * (3.0*L - x) / (6.0*EI) };
            };
            solver.computeL2Error(exact);

            const double W_int = solver.computeInternalEnergy();
            const double W_ext = solver.computeExternalWork();
            const double deltaWrel = abs(W_int - W_ext) / max(abs(W_int), 1e-30);

              cout << "dW_rel=" << deltaWrel << ", L2_rel=" << solver.errL2_rel << endl;

            out = {h, mesh.nbElements(), solver.errL2_rel, deltaWrel};
            return true;
        });

    ReportOptions reportOpt;
    printConvergenceTable(results, "Flexion", reportOpt, [](const ConvergenceResult& r){ return r.L2rel; });
    exportConvergenceCSV(results, "results/convergence_flexion.csv", reportOpt, [](const ConvergenceResult& r){ return r.L2rel; });
}

void runFlexionTest(const string& meshFile, const Config& config) {
    runFlexionTest(vector<string>{meshFile}, vector<double>{1.0}, config);
}
