#include "test_traction.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// TEST TRACTION SIMPLE - CONVERGENCE MAILLAGE
void runTractionTest(const vector<string>& meshFiles,
                     const vector<double>& meshLc,
                     const Config& config) {
    cout << "-------------- TRACTION - CONVERGENCE MAILLAGE --------------" << endl;

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
            applyDistributedForce(solver, mesh, mesh.rightNodes, totalForce, 0);

            solver.applyBC();
            solver.solveConjugateGradient();
            solver.computeStrainStress();
            solver.saveVTK(convergenceVtkPath(config, h));

            const double A = mesh.height();
            const double L = mesh.width();
            const double H = mesh.height();
            const double yMid = 0.5 * (mesh.yMin + mesh.yMax);
            const double FAE  = totalForce / (A * config.E);
            ExactFn exact = [FAE, nu = config.nu, yMid](double x, double y) -> Eigen::Vector2d {
                return { FAE * x, -nu * FAE * (y - yMid) };
            };
            solver.computeL2Error(exact);
            const Eigen::VectorXd U = solver.getU();
            auto meanDisp = [&](const vector<int>& nodes, int dof) {
                double sum = 0.0;
                for (int id : nodes) sum += U(2*(id-1) + dof);
                return sum / max<size_t>(1, nodes.size());
            };

            
            const double sigmaX = totalForce / max(H, 1e-30);
            const double uxRight = meanDisp(mesh.rightNodes, 0);
            const double epsX = uxRight / max(L, 1e-30);
            const double uyTop = meanDisp(mesh.topNodes, 1);
            const double uyBot = meanDisp(mesh.bottomNodes, 1);
            const double epsY = (uyTop - uyBot) / max(H, 1e-30);

            const double Eeff = sigmaX / max(epsX, 1e-30);
            const double nueff = -epsY / max(epsX, 1e-30);
            const double errErel = abs(Eeff - config.E) / max(abs(config.E), 1e-30);
            const double errNurel = abs(nueff - config.nu) / max(abs(config.nu), 1e-30);

            const double W_int = solver.computeInternalEnergy();
            const double W_ext = solver.computeExternalWork();
            const double deltaWrel = abs(W_int - W_ext) / max(abs(W_int), 1e-30);

              cout << "E_eff = " << Eeff << ", nu_eff = " << nueff << ", err_E_rel = " << errErel << ", err_nu_rel = " << errNurel
                   << ", deltaW_rel = " << deltaWrel << ", L2_rel = " << solver.errL2_rel << endl;

            out = {h, mesh.nbElements(), solver.errL2_rel, deltaWrel, errErel, errNurel, -1.0};
            return true;
        });

        ReportOptions reportOpt;
        reportOpt.showL2 = true;
        reportOpt.showDeltaW = true;
        reportOpt.showErrE = true;
        reportOpt.showErrNu = true;
        printConvergenceTable(results, "Traction", reportOpt, [](const ConvergenceResult& r){ return r.L2rel; });
        exportConvergenceCSV(results, "results/convergence_traction.csv", reportOpt, [](const ConvergenceResult& r){ return r.L2rel; });
}
