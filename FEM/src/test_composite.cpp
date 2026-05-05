#include "test_composite.h"
#include "meshReader.h"
#include "tests_utils.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <filesystem>

using namespace std;

static string compositeVtkAlias(const string& meshFile, const string& suffix) {
    namespace fs = std::filesystem;
    const string stem = fs::path(meshFile).stem().string();
    const size_t cut = stem.find_first_of("_-");
    const string prefix = (cut == string::npos) ? stem : stem.substr(0, cut);
    return "results/" + prefix + "_" + suffix + ".vtk";
}

void runCompositeTest(const string& meshFile, const Config& config) {
    // Test composite transverse (original)
    cout << "------------------ TEST COMPOSITE ------------------" << endl;
    cout << "Fichier de maillage: " << meshFile << endl;
    cout << "Nombres de threads: " << omp_get_max_threads() << endl;
    if (config.planTransverse) {


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
    double h = mesh.computeCharacteristicLength();

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
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0);
    for (int id : mesh.findNodesAtY(mesh.yMax / 2.0)) {
        // on regarde si le noeud est bien à gauche
        const Node& node = mesh.getNode(id);
        if (abs(node.coords.x() - mesh.xMin) < 1e-6) {
            solver.setDirichletBC(id, 1, 0.0);
        }
    }
    applyDistributedForce(solver, mesh, mesh.rightNodes, config.forceValue, 0);
    
    solver.applyBC();
    auto start_x = chrono::high_resolution_clock::now();
    solver.solveConjugateGradient();
    auto end_x = chrono::high_resolution_clock::now();
    double tcpu_x = chrono::duration<double>(end_x - start_x).count();
    cout << "Temps de résolution GC (traction x): " << tcpu_x << " s" << endl;
    solver.computeStrainStress();
    solver.saveVTK(compositeVtkAlias(meshFile, "tracx"));
    
    // Calcul critère de rupture Tsai-Hill
    double Xt = 18e6; // Résistance traction fibres (Pa)
    double Yt = 100e6;   // Résistance transverse
    double S = 30e6;    // Résistance cisaillement
    std::vector<double> failureIndex(mesh.nbElements());
    double maxFailure = 0.0;
    for (int i = 0; i < mesh.nbElements(); ++i) {
        failureIndex[i] = solver.computeTsaiHill(i, Xt, Yt, S);
        if (failureIndex[i] > maxFailure) maxFailure = failureIndex[i];
    }
    cout << "Critère de rupture max : " << maxFailure << " (rupture si >1)" << endl;
    
    Eigen::VectorXd U = solver.getU();
    auto calcDisp = [&](const vector<int>& nodes, int dof) {
        double sum = 0.0;
        for (int id : nodes) sum += U(2*(id-1) + dof);
        return sum / max<size_t>(1, nodes.size());
    };
    
    double ux_right = calcDisp(mesh.rightNodes, 0);
    double ux_left = calcDisp(mesh.leftNodes, 0);
    double uy_top = calcDisp(mesh.topNodes, 1);
    double uy_bottom = calcDisp(mesh.bottomNodes, 1);

    double L = mesh.width();
    double H = mesh.height();
    double A_x = H;  // longueur du bord chargé en traction x
    double A_y = L;  // longueur du bord chargé en traction y

    // Déformations moyennes
    double epsilon_x = (ux_right - ux_left) / max(L, 1e-30);
    double epsilon_y = (uy_top - uy_bottom) / max(H, 1e-30);
    
    // Contrainte appliquée
    double sigma_x = config.forceValue / A_x; 
    
    double Wint = solver.computeInternalEnergy();
    double Wext = solver.computeExternalWork();
    double dWrel = abs(Wint - Wext) / max(abs(Wint), 1e-30);
    cout << "\n-------- Propriétés effectives :" << endl;
    comp.updateFromTractionX(sigma_x, epsilon_x, epsilon_y);
    cout << " E_1 : " << comp.E1/1e9 << " GPa" << " (Voigt: " << comp.E1_voigt/1e9 << " GPa, Reuss: " << comp.E1_reuss/1e9 << " GPa, Hill: " << comp.E1_hill/1e9 << " GPa, Halpin-Tsai: " << comp.E1_halpin_tsai/1e9 << " GPa)" << endl;
    cout << " v_12 : " << comp.v12 << " (Voigt: " << comp.v12_voigt << ", Reuss: " << comp.v12_reuss << ", Hill: " << comp.v12_hill << ", Halpin-Tsai: " << comp.v12_halpin_tsai << ")" << endl;
    cout << " Delta W_rel = " << dWrel << endl;

    solver.clearBCs();
    solver.clearSystem();
    cout << "\n-------- Résolution de la traction selon y ---------\n" << endl;
    solver.assemble();
    
    // Encastrement en bas, force en haut
    for (int id : mesh.bottomNodes) {
        solver.setDirichletBC(id, 1, 0.0);
    }

    for (int id : mesh.findNodesAtX(mesh.xMax / 2.0)) {
        // on regarde si le noeud est bien en bas
        const Node& node = mesh.getNode(id);
        if (abs(node.coords.y() - mesh.yMin) < 1e-6) {
            solver.setDirichletBC(id, 0, 0.0);
        }
    }

    applyDistributedForce(solver, mesh, mesh.topNodes, config.forceValue, 1);
    solver.applyBC();
    auto start_y = chrono::high_resolution_clock::now();
    solver.solveConjugateGradient();
    auto end_y = chrono::high_resolution_clock::now();
    double tcpu_y = chrono::duration<double>(end_y - start_y).count();
    cout << "Temps de résolution GC (traction y): " << tcpu_y << " s" << endl;
    solver.computeStrainStress();
    Wint = solver.computeInternalEnergy();
    Wext = solver.computeExternalWork();
    dWrel = abs(Wint - Wext) / max(abs(Wint), 1e-30);
    solver.saveVTK(compositeVtkAlias(meshFile, "tracy"));
    // Calcul de E2 et v21
    U = solver.getU();
    double uy_top_y = calcDisp(mesh.topNodes, 1);
    double uy_bottom_y = calcDisp(mesh.bottomNodes, 1);
    double epsilon_y_y = (uy_top_y - uy_bottom_y) / max(H, 1e-30);
    double epsilon_x_y = (calcDisp(mesh.rightNodes, 0) - calcDisp(mesh.leftNodes, 0)) / max(L, 1e-30);
    double sigma_y = config.forceValue / A_y;
    comp.updateFromTractionY(sigma_y, epsilon_y_y, epsilon_x_y);

    cout << "\n--------   Propriétés effectives :" << endl;
    cout << " E2 : " << comp.E2/1e9 << " GPa" << endl;
    cout << " v21 : " << comp.v21 << endl;

    cout << "\nComparaison symétrie de C :" << endl;
    double C12 = comp.v12 / comp.E1;
    double C21 = comp.v21 / comp.E2; 
    cout << " v21/E2 = " << C21 << ", v12/E1 = " << C12 << "(Delta = " << abs(C21 - C12) << ")" << endl;
    cout << " Delta_W_rel =" << dWrel << endl;
    // Test de cisaillement
    solver.clearBCs();
    solver.clearSystem();

    cout << "\n-------- Résolution du cisaillement ---------\n" << endl;
    solver.assemble();
    
    double gammaTarget = 1.0e-3;
    applyShearDirichletBC(solver, mesh, gammaTarget);
    solver.applyBC();
    auto start_s = chrono::high_resolution_clock::now();
    solver.solveConjugateGradient();
    auto end_s = chrono::high_resolution_clock::now();
    double tcpu_s = chrono::duration<double>(end_s - start_s).count();
    cout << "Temps de résolution GC (cisaillement): " << tcpu_s << " s" << endl;
    solver.computeStrainStress();
    Wint = solver.computeInternalEnergy();
    const double V_shear = max(L * H, 1e-30);
    // Estimation énergétique globale (diagnostic) : sensible aux énergies non-cisaillement.
    const double G12_energy = 2.0 * Wint / max(gammaTarget * gammaTarget * V_shear, 1e-30);
    solver.saveVTK(compositeVtkAlias(meshFile, "shear"));
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
            << " (Voigt: " << comp.G12_voigt/1e9 << " GPa, Reuss: " << comp.G12_reuss/1e9 << " GPa, Hill: " << comp.G12_hill/1e9 << " GPa, Halpin-Tsai: " << comp.G12_halpin_tsai/1e9 << " GPa)" << endl;
    cout << " tau_xy moyen = " << tau_xy/1e6 << " MPa"
         << ", G12_energie = " << G12_energy/1e9 << " GPa" << endl;
    cout << " gamma12 mesuré = " << gamma12 << " (cible: " << gammaTarget << ")" << endl;

    comp.buildMatrixes();
    comp.printC();
    comp.printS();
    comp.printProperties();
    
    // Test G21 : cisaillement appliqué sur le côté droit, encastrement à gauche
    solver.clearBCs();
    solver.clearSystem();
    cout << "\n-------- Résolution du cisaillement G21 (droite encastrement gauche) ---------\n" << endl;
    solver.assemble();
    double gammaTarget21 = 1.0e-3;
    applyShearRightDirichletBC(solver, mesh, gammaTarget21);
    solver.applyBC();
    auto start_g21 = chrono::high_resolution_clock::now();
    solver.solveConjugateGradient();
    auto end_g21 = chrono::high_resolution_clock::now();
    double tcpu_g21 = chrono::duration<double>(end_g21 - start_g21).count();
    cout << "Temps de résolution GC (cisaillement G21): " << tcpu_g21 << " s" << endl;
    solver.computeStrainStress();
    Wint = solver.computeInternalEnergy();
    solver.saveVTK(compositeVtkAlias(meshFile, "shear_g21"));

    // Mesures
    U = solver.getU();
    double uy_right = calcDisp(mesh.rightNodes, 1);
    double uy_left = calcDisp(mesh.leftNodes, 1);
    double gamma21 = (uy_right - uy_left) / max(L, 1e-30);

    // Calcul robuste: contrainte de cisaillement moyenne issue du champ FE.
    const Eigen::MatrixXd stress_g21 = solver.getStress();
    double sumTauA_g21 = 0.0;
    double sumArea_g21 = 0.0;
    for (int e = 0; e < mesh.nbElements(); ++e) {
        const double area = mesh.elements[e]->area;
        sumTauA_g21 += stress_g21(e, 2) * area;
        sumArea_g21 += area;
    }
    const double tau_yx = sumTauA_g21 / max(sumArea_g21, 1e-30);

    // Estimation énergétique globale
    const double V_shear_g21 = max(L * H, 1e-30);
    const double G21_energy = 2.0 * Wint / max(gammaTarget21 * gammaTarget21 * V_shear_g21, 1e-30);

    double G21_fem = tau_yx / max(gamma21, 1e-30);
    cout << "\n-------- Propriétés cisaillement G21 :" << endl;
    cout << " G21_energy = " << G21_energy/1e9 << " GPa" << endl;
    cout << " G21_FEM = " << G21_fem/1e9 << " GPa" << endl;
    cout << " tau_yx moyen = " << tau_yx/1e6 << " MPa" << endl;
    cout << " gamma21 mesuré = " << gamma21 << " (cible: " << gammaTarget21 << ")" << endl;

    // Export CSV pour G21
    string meshNameG = meshFile;
    size_t startG = meshNameG.find_last_of('/') + 1;
    size_t endG = meshNameG.find_last_of('.');
    if (endG == string::npos) endG = meshNameG.size();
    meshNameG = meshNameG.substr(startG, endG - startG);
    string csvG21 = "results/proprietes_" + meshNameG + "_G21.csv";
    ofstream csv2(csvG21);
    if (csv2.is_open()) {
        csv2 << fixed << setprecision(8);
        csv2 << "Property,Value\n";
        csv2 << "G21_energy," << G21_energy << "\n";
        csv2 << "G21_FEM," << G21_fem << "\n";
        csv2 << "tau_yx," << tau_yx << "\n";
        csv2 << "gamma21," << gamma21 << "\n";
        csv2 << "tcpug21," << tcpu_g21 << "\n";
        csv2.close();
        cout << "Fichier CSV G21 exporté : " << csvG21 << "\n" << endl;
    } else {
        cerr << "Impossible d'écrire: " << csvG21 << "\n";
    }
    // Export des propriétés effectives vers CSV
    string meshName = meshFile;
    size_t start = meshName.find_last_of('/') + 1;
    size_t end = meshName.find_last_of('.');
    if (end == string::npos) end = meshName.size();
    meshName = meshName.substr(start, end - start);
    string csvFile = "results/proprietes_" + meshName + ".csv";
    double tcpumax = max({tcpu_x, tcpu_y, tcpu_s});
    exportCompositePropertiesCSV(comp, csvFile, h, mesh.nbElements(), tcpumax);
    }

    if (!config.planTransverse) {
        // Test 1 : Traction longitudinale pour extraire E3
        cout << "------------------ TEST 1 - TRACTION LONGITUDINALE ------------------" << endl;
        cout << "Fichier de maillage: " << meshFile << endl;
        cout << "Nombres de threads: " << omp_get_max_threads() << endl;

        Mesh mesh;
        Material matrix(config.E, config.nu, config.rho);
        Material fiber(config.E_fiber, config.nu_fiber, config.rho_fiber);
        Material pore(config.E_pore, config.nu_pore, config.rho_pore);
        CompositeMaterial comp(&matrix, &fiber, config.hasPores ? &pore : &matrix);
        MeshReader reader(&mesh);
        reader.setMaterial(1, &matrix);
        reader.setMaterial(2, &fiber);
        reader.setMaterial(3, config.hasPores ? &pore : &matrix);
        reader.readGmshFile(meshFile);
        mesh.initializeElements();
        mesh.computeGeometry();
        double h = mesh.computeCharacteristicLength();

        const double V_fiber = mesh.computeVolumeFraction(comp.fiber);
        const double V_pore = config.hasPores ? mesh.computeVolumeFraction(comp.pore) : 0.0;
        const double V_matrix = max(0.0, 1.0 - V_fiber - V_pore);
        comp.setVolumeFractions(V_fiber, V_matrix, V_pore);
        comp.computeEffectiveProperties();

        cout << " Volume fraction : Fiber = " << comp.V_fiber << ", Matrix = " << comp.V_matrix;
        if (config.hasPores) { cout << ", Pore = " << comp.V_pore; }
        cout << endl;

        // Résolution de la traction longitudinale (selon direction longitudinale)
        cout << "\n-------- Résolution de la traction longitudinale ---------\n" << endl;
        Solver solver(mesh, config.solverTolerance, config.solverMaxIter);
        applySolverConfig(solver, config);
        solver.assemble();
        
        for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0);
        for (int id : mesh.findNodesAtY(mesh.yMax / 2.0)) {
            const Node& node = mesh.getNode(id);
            if (abs(node.coords.x() - mesh.xMin) < 1e-6) {
                solver.setDirichletBC(id, 1, 0.0);
            }
        }
        applyDistributedForce(solver, mesh, mesh.rightNodes, config.forceValue, 0);
        
        solver.applyBC();
        auto start_long = chrono::high_resolution_clock::now();
        solver.solveConjugateGradient();
        auto end_long = chrono::high_resolution_clock::now();
        double tcpu_long = chrono::duration<double>(end_long - start_long).count();
        cout << "Temps de résolution GC (traction longitudinale): " << tcpu_long << " s" << endl;
        solver.computeStrainStress();
        solver.saveVTK(compositeVtkAlias(meshFile, "traclong"));
        
        // Calcul de E3 à partir de la traction longitudinale
        Eigen::VectorXd U = solver.getU();
        auto calcDisp = [&](const vector<int>& nodes, int dof) {
            double sum = 0.0;
            for (int id : nodes) sum += U(2*(id-1) + dof);
            return sum / max<size_t>(1, nodes.size());
        };
        
        double uy_top = calcDisp(mesh.topNodes, 1);
        double uy_bottom = calcDisp(mesh.bottomNodes, 1);
        double ux_right = calcDisp(mesh.rightNodes, 0);
        double ux_left = calcDisp(mesh.leftNodes, 0);
        
        double H = mesh.height();
        double L = mesh.width();

        // Déformation longitudinale
        double epsilon_long = (ux_right - ux_left) / max(L, 1e-30);
        double epsilon_trans = (uy_top - uy_bottom) / max(H, 1e-30);
        
        // Contrainte appliquée
        double sigma_long = config.forceValue / max(H, 1e-30);
        
        // Calcul de E3
        const double eps = 1e-30;
        double E3 = sigma_long / max(abs(epsilon_long), eps);
        double nu = -epsilon_trans / max(abs(epsilon_long), eps);

        // Bornes analytiques Voigt/Reuss/Hill en longitudinal
        double E3_voigt = V_fiber * fiber.E + V_matrix * matrix.E + V_pore * pore.E;
        double E3_reuss = 1.0 / max(V_fiber / max(fiber.E, eps)
                      + V_matrix / max(matrix.E, eps)
                      + V_pore / max(pore.E, eps), eps);
        double E3_hill = 0.5 * (E3_voigt + E3_reuss);

        double nu_voigt = V_fiber * fiber.nu + V_matrix * matrix.nu + V_pore * pore.nu;
        double nu_reuss = 1.0 / max(V_fiber / max(fiber.nu, eps)
                      + V_matrix / max(matrix.nu, eps)
                      + V_pore / max(pore.nu, eps), eps);
        double nu_hill = 0.5 * (nu_voigt + nu_reuss);

        double Wint = solver.computeInternalEnergy();
        double Wext = solver.computeExternalWork();
        double dWrel = abs(Wint - Wext) / max(abs(Wint), 1e-30);
        
        cout << "\n-------- Propriétés effectives :" << endl;
           cout << " E3 : " << E3/1e9 << " GPa"
               << " (Voigt: " << E3_voigt/1e9 << " GPa, Reuss: " << E3_reuss/1e9 << " GPa, Hill: " << E3_hill/1e9 << " GPa)" << endl;
           cout << " nu : " << nu
               << " (Voigt: " << nu_voigt << ", Reuss: " << nu_reuss << ", Hill: " << nu_hill << ")" << endl;
        cout << " Déformation longitudinale : " << epsilon_long << endl;
           cout << " Déformation transverse : " << epsilon_trans << endl;
        cout << " Contrainte appliquée : " << sigma_long/1e6 << " MPa" << endl;
        cout << " Delta W_rel = " << dWrel << endl;

        // Export des propriétés effectives vers CSV
        string meshName = meshFile;
        size_t start = meshName.find_last_of('/') + 1;
        size_t end = meshName.find_last_of('.');
        if (end == string::npos) end = meshName.size();
        meshName = meshName.substr(start, end - start);
        string csvFile = "results/proprietes_" + meshName + "_E3.csv";
        ofstream csv(csvFile);
        if (!csv.is_open()) { cerr << "Impossible d'écrire: " << csvFile << "\n"; return; }
        csv << fixed << setprecision(8);
        csv << "Property,Voigt,Reuss,Hill\n";
        csv << "h," << h << ",,\n";
        csv << "E3," << E3_voigt << "," << E3_reuss << "," << E3_hill << "\n";
        csv << "E3_FEM," << E3 << ",,\n";
        csv << "nu," << nu_voigt << "," << nu_reuss << "," << nu_hill << "\n";
        csv << "nu_FEM," << nu << ",,\n";
        csv << "epsilon_long," << epsilon_long << ",,\n";
        csv << "epsilon_trans," << epsilon_trans << ",,\n";
        csv << "sigma_long," << sigma_long << ",,\n";
        csv << "V_fiber," << comp.V_fiber << ",,\n";
        csv << "V_matrix," << comp.V_matrix << ",,\n";
        csv << "V_pore," << comp.V_pore << ",,\n";
        csv << "tcpumax," << tcpu_long << ",,\n";
        csv.close();
        cout << "Fichier CSV des propriétés exporté : " << csvFile << "\n" << endl;
    }
}
