#include "test_flexion.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// TEST FLEXION LINEAIRE - CONVERGENCE MAILLAGE
// Traction linéaire sur le bord droit (direction y) :
// L2 est évalué via une solution analytique de poutre (Euler-Bernoulli)
// en superposant la contribution de la force de bout F et du moment de bout M.
void runFlexionTest(const vector<string>& meshFiles, const vector<double>& meshLc, const Config& config) {
    cout << "---------- FLEXION LINEAIRE - CONVERGENCE MAILLAGE ----------" << endl;

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

            const double L    = mesh.width();
            const double H    = mesh.height();
            const double I    = H * H * H / 12.0;
            const double yMin = mesh.yMin;
            const double yMid = 0.5 * (mesh.yMin + mesh.yMax);
            const double F    = config.forceValue;

            const double a = 2.0 * F / max(H * H, 1e-30);

            
            applyDistributedForce(solver, mesh, mesh.rightNodes,
                [a, yMin](double y) {
                    return a * (y - yMin);
                }, 1);

            solver.applyBC();
            solver.solveConjugateGradient();
            solver.computeStrainStress();
            solver.saveVTK(convergenceVtkPath(config, h));

            // Résultante et moment de bout équivalents du profil linéaire
            const double F_eq = F;
            const double M_eq = F * H / 6.0;
            const double EI   = config.E * I;

            ExactFn exact = [F_eq, M_eq, L, EI, yMid](double x, double y) -> Eigen::Vector2d {
                const double yn = y - yMid;
                const double theta = (F_eq * x * (2.0 * L - x) / (2.0 * EI))
                                   + (M_eq * x / EI);
                const double v = (F_eq * x * x * (3.0 * L - x) / (6.0 * EI))
                               + (M_eq * x * x / (2.0 * EI));
                return { -yn * theta, v };
            };
            solver.computeL2Error(exact);

            const double W_int    = solver.computeInternalEnergy();
            const double W_ext    = solver.computeExternalWork();
            const double deltaWrel = abs(W_int - W_ext) / max(abs(W_int), 1e-30);

            cout << "  dW_rel=" << deltaWrel
                 << ", L2_rel=" << solver.errL2_rel << endl;

            out = {h, mesh.nbElements(), solver.errL2_rel, deltaWrel};
            return true;
        });

    ReportOptions reportOpt;
    reportOpt.showL2     = true;
    reportOpt.showDeltaW = true;
    printConvergenceTable(results, "Flexion Lineaire", reportOpt,
        [](const ConvergenceResult& r){ return r.L2rel; });
    exportConvergenceCSV(results, "results/convergence_flexion.csv", reportOpt,
        [](const ConvergenceResult& r){ return r.L2rel; });
}

void runFlexionTest(const string& meshFile, const Config& config) {
    runFlexionTest(vector<string>{meshFile}, vector<double>{1.0}, config);
}
