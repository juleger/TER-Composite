#include "test_composite.h"
#include "meshReader.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>

using namespace std;

void runCompositeTest(const string& meshFile, const Config& config) {
    cout << "------------------ TEST COMPOSITE ------------------" << endl;
    cout << "Fichier de maillage: " << meshFile << endl;
    cout << "Nombres de threads: " << omp_get_max_threads() << endl;


    Mesh mesh;
    Material matrix(config.E, config.nu, config.rho);
    Material fiber(config.E_fiber, config.nu_fiber, config.rho_fiber);
    Material pore(config.E_pore, config.nu_pore, config.rho_pore);
    CompositeMaterial comp(&matrix, &fiber, config.hasPores ? &pore : &matrix);
    MeshReader reader(&mesh);
    reader.setMaterial(1, &matrix);  // Matériau 1 = matrice
    reader.setMaterial(2, &fiber);   // Matériau 2 = fibre
    reader.setMaterial(3, config.hasPores ? &pore : &matrix); // Matériau 3 = pores (si présents) ou matrice sinon
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();

        const double V_fiber = mesh.computeVolumeFraction(comp.fiber);
        const double V_pore = config.hasPores ? mesh.computeVolumeFraction(comp.pore) : 0.0;
        const double V_matrix = max(0.0, 1.0 - V_fiber - V_pore);
        comp.setVolumeFractions(V_fiber, V_matrix, V_pore);
        comp.computeEffectiveProperties();

        cout << " Volume fraction : Fiber = " << comp.V_fiber << ", Matrix = " << comp.V_matrix;
        if (config.hasPores) { cout << ", Pore = " << comp.V_pore; }
        cout << endl;

    // Résolution

    cout << "-------- Résolution de la traction selon x ---------\n" << endl;
    Solver solver(mesh, config.solverTolerance, config.solverMaxIter);
    applySolverConfig(solver, config);
    solver.assemble();
    
    // Conditions aux limites: encastrement à gauche, force à droite
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0), solver.setDirichletBC(id, 1, 0.0);
    applyDistributedForce(solver, mesh, mesh.rightNodes, config.forceValue, 0);
    
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.computeStrainStress();
    solver.saveVTK("results/composite_traction_x.vtk");
    
    Eigen::VectorXd U = solver.getU();
    auto calcDisp = [&](const vector<int>& nodes, int dof) {
        double sum = 0.0;
        for (int id : nodes) sum += U(2*(id-1) + dof);
        return sum / max<size_t>(1, nodes.size());
    };
    
    double ux = calcDisp(mesh.rightNodes, 0);
    double uy = (abs(calcDisp(mesh.topNodes, 1)) + abs(calcDisp(mesh.bottomNodes, 1))) / 2.0;

    double L = mesh.width();
    double H = mesh.height();
    double A = H;  // section transversale (épaisseur unité = 1)

    // Déformations moyennes
    double epsilon_x = ux / L;
    double epsilon_y = -uy / (H/2.0);
    
    // Contrainte appliquée
    double sigma_x = config.forceValue / A; 
    
    double Wint = solver.computeInternalEnergy();
    double Wext = solver.computeExternalWork();
    double dWrel = abs(Wint - Wext) / max(abs(Wint), 1e-30);
    cout << "\n-------- Propriétés effectives :" << endl;
    comp.updateFromTractionX(sigma_x, epsilon_x, epsilon_y);
    cout << " E_1 : " << comp.E1/1e9 << " GPa" << " (Voigt: " << comp.E1_voigt/1e9 << " GPa, Reuss: " << comp.E1_reuss/1e9 << " GPa)" << endl;
    cout << " v_12 : " << comp.v12 << " (Voigt: " << comp.v12_voigt << ", Reuss: " << comp.v12_reuss << ")" << endl;
    cout << " Delta W_rel = " << dWrel << endl;

    solver.clearBCs();
    solver.clearSystem();
    cout << "\n-------- Résolution de la traction selon y ---------\n" << endl;
    solver.assemble();
    
    // Encastrement en bas, force en haut
    for (int id : mesh.bottomNodes) {
        solver.setDirichletBC(id, 0, 0.0);
        solver.setDirichletBC(id, 1, 0.0);
    }

    applyDistributedForce(solver, mesh, mesh.topNodes, config.forceValue, 1);
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.computeStrainStress();
    Wint = solver.computeInternalEnergy();
    Wext = solver.computeExternalWork();
    dWrel = abs(Wint - Wext) / max(abs(Wint), 1e-30);
    solver.saveVTK("results/composite_traction_y.vtk");
    // Calcul de E2 et v21
    U = solver.getU();
    double uy_y = calcDisp(mesh.topNodes, 1);
    double epsilon_y_y = uy_y / H;
    double epsilon_x_y = calcDisp(mesh.rightNodes, 0) / max(L, 1e-30);
    double sigma_y = config.forceValue / A;
    comp.updateFromTractionY(sigma_y, epsilon_y_y, epsilon_x_y);

    cout << "\n--------   Propriétés effectives :" << endl;
    cout << " E2 : " << comp.E2/1e9 << " GPa" << endl;
    cout << " v21 : " << comp.v21 << endl;

    cout << "\nComparaison symétrie de C :" << endl;
    double C12 = comp.v12 / comp.E2;
    double C21 = comp.v21 / comp.E1; 
    cout << " v21/E1 = " << C21 << ", v12/E2 = " << C12 << "(Delta = " << abs(C21 - C12) << ")" << endl;
    cout << " Delta_W_rel =" << dWrel << endl;
    // Test de cisaillement
    solver.clearBCs();
    solver.clearSystem();

    cout << "\n-------- Résolution du cisaillement ---------\n" << endl;
    solver.assemble();
    
    double gammaTarget = 1.0e-3;
    applyShearDirichletBC(solver, mesh, gammaTarget);
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.computeStrainStress();
    Wint = solver.computeInternalEnergy();
    const double V_shear = max(L * H, 1e-30);
    // Estimation énergétique globale (diagnostic) : sensible aux énergies non-cisaillement.
    const double G12_energy = 2.0 * Wint / max(gammaTarget * gammaTarget * V_shear, 1e-30);
    solver.saveVTK("results/composite_shear.vtk");
    U = solver.getU();
    // Déformation relative mesurée
    double gamma12 = (calcDisp(mesh.topNodes, 0) - calcDisp(mesh.bottomNodes, 0)) / max(H, 1e-30);
    // Calcul robuste: contrainte de cisaillement moyenne issue du champ FE.
    const Eigen::MatrixXd stress = solver.getStress();
    double sumTauA = 0.0;
    double sumArea = 0.0;
    for (int e = 0; e < mesh.nbElements(); ++e) {
        const double area = mesh.elements[e]->area;
        sumTauA += stress(e, 2) * area;
        sumArea += area;
    }
    const double tau_xy = sumTauA / max(sumArea, 1e-30);
    comp.updateFromShear(tau_xy, gamma12);
    cout << "\n-------- Propriétés effectives :" << endl;
        cout << " G12 : " << comp.G12/1e9 << " GPa"
            << " (Voigt: " << comp.G12_voigt/1e9 << " GPa, Reuss: " << comp.G12_reuss/1e9 << " GPa)" << endl;
    cout << " tau_xy moyen = " << tau_xy/1e6 << " MPa"
         << ", G12_energie = " << G12_energy/1e9 << " GPa" << endl;
    cout << " gamma12 mesuré = " << gamma12 << " (cible: " << gammaTarget << ")" << endl;

    comp.buildMatrixes();
    comp.printC();
    comp.printS();
}
