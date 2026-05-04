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
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0), solver.setDirichletBC(id, 1, 0.0);
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
        solver.setDirichletBC(id, 0, 0.0);
        solver.setDirichletBC(id, 1, 0.0);
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
        // Test longitudinal : estimations analytiques pour E3
        cout << "------------------ TEST LONGITUDINAL ------------------" << endl;
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

        // Estimations analytiques pour E3 (direction transverse/longitudinale)
        double E3_voigt = V_fiber * fiber.E + V_matrix * matrix.E + V_pore * pore.E;
        double E3_reuss = 1.0 / (V_fiber / fiber.E + V_matrix / matrix.E + V_pore / pore.E);
        double E3_hill = 0.5 * (E3_voigt + E3_reuss);

        cout << "\n-------- Propriétés effectives longitudinales :" << endl;
        cout << " E3 : " << E3_voigt/1e9 << " GPa (Voigt), " << E3_reuss/1e9 << " GPa (Reuss), " << E3_hill/1e9 << " GPa (Hill)" << endl;

        // Export CSV
        string meshName = meshFile;
        size_t start = meshName.find_last_of('/') + 1;
        size_t end = meshName.find_last_of('.');
        if (end == string::npos) end = meshName.size();
        meshName = meshName.substr(start, end - start);
        string csvFile = "results/proprietes_longitudinal_" + meshName + ".csv";
        ofstream csv(csvFile);
        if (!csv.is_open()) { cerr << "Impossible d'écrire: " << csvFile << "\n"; return; }
        csv << fixed << setprecision(8);
        csv << "Property,Voigt,Reuss,Hill\n";
        csv << "h," << h << ",,\n";
        csv << "E3," << E3_voigt << "," << E3_reuss << "," << E3_hill << "\n";
        csv << "V_fiber," << comp.V_fiber << ",,\n";
        csv << "V_matrix," << comp.V_matrix << ",,\n";
        csv << "V_pore," << comp.V_pore << ",,\n";
        csv << "tcpumax,0.0,\n";
        csv.close();
        cout << "Fichier CSV des propriétés longitudinales exporté : " << csvFile << "\n" << endl;
    }
}
