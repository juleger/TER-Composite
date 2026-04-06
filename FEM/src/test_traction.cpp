#include "test_traction.h"
#include "tests_utils.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// TEST TRACTION GAUSSIENNE - CONVERGENCE MAILLAGE
// Force distribuée sur le bord droit selon un profil gaussien en y :
// f(y) = A * exp(-(y - y_mid)^2 / (2*sigma^2))
// Pas de solution analytique simple : on valide via le bilan énergétique (delta W).
void runTractionTest(const vector<string>& meshFiles,const vector<double>& meshLc, const Config& config) {
    cout << "--------- TRACTION GAUSSIENNE - CONVERGENCE MAILLAGE ---------" << endl;
    Material material(config.E, config.nu, config.rho);

    vector<ConvergenceResult> results = runConvergence(
        meshFiles, meshLc,
        [&](size_t i, const string& mfile, double h, ConvergenceResult& out) {
            Mesh mesh;
            loadAndInitMesh(mfile, material, mesh);

            Solver solver(mesh, config.solverTolerance, config.solverMaxIter);
            applySolverConfig(solver, config);
            solver.assemble();

            for (int id : mesh.leftNodes)
                solver.setDirichletBC(id, 0, 0.0);
            for (int id : mesh.findNodesAtY(mesh.yMax / 2.0))
                solver.setDirichletBC(id, 1, 0.0);

            const double totalForce = config.forceValue;
            const double yMid  = 0.5 * (mesh.yMin + mesh.yMax);
            const double H     = mesh.height();
            const double sigma = H / 4.0;

            const double sq2          = std::sqrt(2.0);
            const double gaussIntegral = sigma * std::sqrt(2.0 * M_PI) * (std::erf((mesh.yMax - yMid) 
                / (sigma * sq2)) - std::erf((mesh.yMin - yMid) / (sigma * sq2))) / 2.0;
            const double A = totalForce / gaussIntegral;

            

            applyDistributedForce(solver, mesh, mesh.rightNodes,
                [A, yMid, sigma](double y) {
                    return A * std::exp(-((y - yMid) * (y - yMid)) / (2.0 * sigma * sigma));
                }, 0);

            solver.applyBC();
            solver.solveConjugateGradient();
            solver.computeStrainStress();
            solver.saveVTK(convergenceVtkPath(config, h));

            const double L = mesh.width();
            const double H_mesh = mesh.height();
            const Eigen::VectorXd U = solver.getU();
            auto meanDisp = [&](const vector<int>& nodes, int dof) {
                double sum = 0.0;
                for (int id : nodes) sum += U(2*(id-1) + dof);
                return sum / max<size_t>(1, nodes.size());
            };

            const double sigmaX  = totalForce / max(H_mesh, 1e-30);
            const double uxRight = meanDisp(mesh.rightNodes, 0);
            const double epsX    = uxRight / max(L, 1e-30);
            const double uyTop   = meanDisp(mesh.topNodes, 1);
            const double uyBot   = meanDisp(mesh.bottomNodes, 1);
            const double epsY    = (uyTop - uyBot) / max(H_mesh, 1e-30);

            const double Eeff    = sigmaX / max(epsX, 1e-30);
            const double nueff   = -epsY  / max(epsX, 1e-30);
            const double errErel  = abs(Eeff  - config.E)   / max(abs(config.E),   1e-30);
            const double errNurel = abs(nueff - config.nu)  / max(abs(config.nu),  1e-30);

            const double W_int    = solver.computeInternalEnergy();
            const double W_ext    = solver.computeExternalWork();
            const double deltaWrel = abs(W_int - W_ext) / max(abs(W_int), 1e-30);

            cout << "  E_eff = " << Eeff << ", nu_eff = " << nueff
                 << ", err_E_rel = " << errErel << ", err_nu_rel = " << errNurel
                 << ", deltaW_rel = " << deltaWrel << endl;

            out = {h, mesh.nbElements(), -1.0, deltaWrel, errErel, errNurel, -1.0};
            return true;
        });

    ReportOptions reportOpt;
    reportOpt.showL2     = false;
    reportOpt.showDeltaW = true;
    reportOpt.showErrE   = true;
    reportOpt.showErrNu  = true;
    printConvergenceTable(results, "Traction Gaussienne", reportOpt,
        [](const ConvergenceResult& r){ return r.deltaWrel; });
    exportConvergenceCSV(results, "results/convergence_traction.csv", reportOpt,
        [](const ConvergenceResult& r){ return r.deltaWrel; });
}
