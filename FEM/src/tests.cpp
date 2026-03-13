#include "tests.h"
#include "mesh.h"
#include "material.h"
#include "solver.h"
#include "meshReader.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

// Fonction utilitaire pour appliquer une force répartie sur une face quelconque du maillage (direction normale)
void applyDistributedForce(Solver& solver, const Mesh& mesh, const vector<int>& nodeIds, double totalForce, int dof) {
    vector<pair<int, double>> nodesY;
    for (int id : nodeIds) {
        nodesY.push_back({id, mesh.getNode(id).coords.y()});
    }
    sort(nodesY.begin(), nodesY.end(), 
         [](const pair<int,double>& a, const pair<int,double>& b) { return a.second < b.second; });
    
    for (size_t i = 0; i < nodesY.size(); i++) {
        double len;
        if (i == 0) {
            len = (nodesY[1].second - nodesY[0].second) / 2.0;
        } else if (i == nodesY.size() - 1) {
            len = (nodesY[i].second - nodesY[i-1].second) / 2.0;
        } else {
            len = (nodesY[i+1].second - nodesY[i-1].second) / 2.0;
        }
        solver.setNeumannBC(nodesY[i].first, dof, totalForce * len / mesh.height());
    }
}

// Fonction pour calculer l'allongement selon une direction donnée

double calcExtension(const Mesh& mesh, const Eigen::VectorXd& U, const vector<int>& nodes, int dof) {
    double sum = 0;
    for (int id : nodes) {
        sum += U(2*(id-1) + dof);
    }
    return sum / nodes.size();
}

// TEST TRACTION SIMPLE
void runTractionTest(const string& meshFile, const Config& config) {
    cout << "=== Test de Traction Simple ===" << endl;
    cout << "Maillage: " << meshFile << endl;
    
    Material material(config.E, config.nu, config.rho);
    
    Mesh mesh;
    MeshReader reader(&mesh);
    reader.setMaterial(1, &material);
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();
    mesh.scaleCoordinates();

    Solver solver(mesh);
    solver.assemble();
    
    // CL: encastrement à gauche, force à droite
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0);
    for (int id : mesh.findNodesAtY(mesh.yMax / 2.0)) solver.setDirichletBC(id, 1, 0.0);
    
    // Force répartie à droite
    double totalForce = config.forceValue;
    applyDistributedForce(solver, mesh, mesh.rightNodes, totalForce, 0);

    solver.applyBC();
    solver.solveConjugateGradient();
    solver.saveVTK(config.outputFilePrefix + ".vtk");

    // Validation avec résultats théoriques
    Eigen::VectorXd U = solver.getU();
    double ux = calcExtension(mesh, U, mesh.rightNodes, 0);
    double uy = (abs(calcExtension(mesh, U, mesh.topNodes, 1)) + abs(calcExtension(mesh, U, mesh.bottomNodes, 1))) / 2.0;
    
    double A = mesh.height(), L = mesh.width();
    // Delta L = FL/AE
    double ux_theo = totalForce * L / (A * config.E);
    double uy_theo = config.nu * totalForce * mesh.height() / (2.0 * A * config.E);
    
    cout << "\n=== Résultats ===" << endl;
    cout << "Allongement x: " << ux << " m (théo: " << ux_theo << ", erreur: " 
         << abs(ux-ux_theo)/ux_theo*100 << "%)" << endl;
    cout << "Contraction y: " << uy << " m (théo: " << uy_theo << ", erreur: " 
         << abs(uy-uy_theo)/uy_theo*100 << "%)" << endl;

    // Solution exacte : u_x = (F/AE)*x,  u_y = -(nu*F/AE)*y
    double FAE = totalForce / (A * config.E);
    ExactFn exact_traction = [FAE, nu = config.nu](double x, double y) -> Eigen::Vector2d {
        return { FAE * x, -nu * FAE * y };
    };
    solver.computeL2Error(exact_traction);
    cout << "Erreur L2(u) : " << solver.errL2 << " m" << endl;
    cout << "Erreur L2(u) relative : " << solver.errL2_rel * 100 << "%" << endl;
}


// TEST FLEXION SIMPLE
void runFlexionTest(const string& meshFile, const Config& config) {
    cout << "=== Test de Flexion (force ponctuelle) ===" << endl;
    cout << "Maillage: " << meshFile << endl;
    
    Material material(config.E, config.nu, config.rho);
    
    Mesh mesh;
    MeshReader reader(&mesh);
    reader.setMaterial(1, &material);
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();
    mesh.scaleCoordinates();  // Appliquer le facteur d'échelle
    
    
    Solver solver(mesh, 1e-6, 2000);
    solver.assemble();
    
    // Encastrement à gauche (uy = 0)
    for (int id : mesh.leftNodes) {
        solver.setDirichletBC(id, 0, 0.0);
        solver.setDirichletBC(id, 1, 0.0);
    }
    
    // Force ponctuelle à l'extrémité droite (au milieu en hauteur)
    int nodeForce = -1;
    double targetY = mesh.yMax / 2.0;
    double minDist = 1e10;
    
    for (int id : mesh.rightNodes) {
        double dist = abs(mesh.getNode(id).coords.y() - targetY);
        if (dist < minDist) {
            minDist = dist;
            nodeForce = id;
        }
    }
    
    double F = config.forceValue;
    solver.setNeumannBC(nodeForce, 1, F);
    
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.saveVTK(config.outputFilePrefix + ".vtk");
    
    // Flèche au point d'application de la force
    Eigen::VectorXd U = solver.getU();
    double fleche = abs(U(2*(nodeForce-1) + 1));
    
    // Calcul de l'erreur L2 sur la flèche sur l'ensemble de la poutre
    double errorL2 = 0.0;
    for (int id : mesh.rightNodes) {
        double uy = abs(U(2*(id-1) + 1));
        errorL2 += uy * uy;
    }
    // Flèche théorique poutre encastrée avec force ponctuelle à l'extrémité:
    // ymax = (F × L³) / (3 × E × I)
    double L = mesh.width();
    double h = mesh.height();
    double I = (1.0 * h * h * h) / 12.0;
    double fleche_theo = (abs(F) * L * L * L) / (3.0 * config.E * I);
    
    cout << "\n=== Résultats ===" << endl;
    cout << "Flèche à l'extrémité: " << fleche << " m" << endl;
    cout << "Flèche théorique: " << fleche_theo << " m" << endl;
    cout << "Erreur relative: " << abs(fleche - fleche_theo) / fleche_theo * 100 << "%" << endl;

    // Solution exacte Euler-Bernoulli
    double EI = config.E * I;
    ExactFn exact_flexion = [F, L, EI, h](double x, double y) -> Eigen::Vector2d {
        double yNeutral = y - h / 2.0;
        double uy_eb = F * x*x * (3.0*L - x) / (6.0*EI);
        double ux_eb = -F * x * yNeutral * (2.0*L - x) / (2.0*EI);
        return { ux_eb, uy_eb };
    };
    solver.computeL2Error(exact_flexion);
    cout << "Erreur L2(u) : " << solver.errL2 << " m" << endl;
    cout << "Erreur L2(u) relative : " << solver.errL2_rel * 100 << "%" << endl;
}


// TEST CISAILLEMENT PUR
void runShearTest(const string& meshFile, const Config& config) {
    cout << "=== Test de Cisaillement ===" << endl;
    cout << "Maillage: " << meshFile << endl;
    
    Material material(config.E, config.nu, config.rho);
    
    Mesh mesh;
    MeshReader reader(&mesh);
    reader.setMaterial(1, &material);
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();
    mesh.scaleCoordinates();
    
    // À implémenter

}

void runCompositeTest(const string& meshFile, const Config& config) {
    cout << "=== Test Composite ===" << endl;
    cout << "Maillage: " << meshFile << endl;
    
    // Créer les deux matériaux
    Material matrix(config.E, config.nu, config.rho);
    Material fiber(config.E_fiber, config.nu_fiber, config.rho_fiber);
    Material pore(config.E_pore, config.nu_pore, config.rho_pore);

    CompositeMaterial comp;
    
    // Charger le maillage
    Mesh mesh;
    MeshReader reader(&mesh);
    reader.setMaterial(1, &matrix);  // Matériau 1 = matrice
    reader.setMaterial(2, &fiber);   // Matériau 2 = fibre
    // Le tag 3 correspond aux pores dans les maillages composites poro-structurés.
    // Sans paramètres de pore explicites, on retombe sur la matrice.
    reader.setMaterial(3, config.hasPores ? &pore : &matrix);
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();
    mesh.scaleCoordinates();  // Appliquer le facteur d'échelle
    double V_fiber = mesh.computeVolumeFraction(&fiber);
    double V_pore = config.hasPores ? mesh.computeVolumeFraction(&pore) : 0.0;
    double V_matrix = max(0.0, 1.0 - V_fiber - V_pore);

    // Résolution
    Solver solver(mesh, 1e-6, 5000);
    solver.assemble();
    
    // Conditions aux limites: encastrement à gauche, force à droite
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0), solver.setDirichletBC(id, 1, 0.0);
    
    // Force répartie à droite
    double totalForce = config.forceValue;
    applyDistributedForce(solver, mesh, mesh.rightNodes, totalForce, 0);
    
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.saveVTK(config.outputDir + "/" + config.outputFilePrefix + ".vtk");
    
    // Résultats
    Eigen::VectorXd U = solver.getU();
    auto calcDisp = [&](const vector<int>& nodes, int dof) {
        double sum = 0;
        for (int id : nodes) sum += U(2*(id-1) + dof);
        return sum / nodes.size();
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
    double sigma_x = totalForce / A; 
    
    // Propriétés effectives homogénéisées
    comp.E1 = sigma_x / epsilon_x;            // Module de Young effectif
    comp.v12 = -epsilon_y / epsilon_x;        // Coefficient de Poisson effectif
    
    double E1_voigt = V_fiber * config.E_fiber + V_matrix * config.E + V_pore * config.E_pore;
    double v12_voigt = V_fiber * config.nu_fiber + V_matrix * config.nu + V_pore * config.nu_pore;

    double E1_reuss = 1.0 / (V_fiber / config.E_fiber + V_matrix / config.E + V_pore / config.E_pore);
    double v12_reuss = 1.0 / (V_fiber / config.nu_fiber + V_matrix / config.nu + V_pore / config.nu_pore);
    cout << "\n=== Résultats ===" << endl;
    cout << "Déplacements : ux = " << ux << " m, uy = " << uy << " m" << endl;
    
    cout << "\n=== Propriétés effectives du composite ===" << endl;
    cout << "  E_1 : " << comp.E1/1e9 << " GPa" << " (Voigt: " << E1_voigt/1e9 << " GPa, Reuss: " << E1_reuss/1e9 << " GPa)" << endl;
    cout << "  v_12 : " << comp.v12 << " (Voigt: " << v12_voigt << ", Reuss: " << v12_reuss << ")" << endl;
    cout << "  Volume fraction matrice : " << V_matrix*100 << " %" << endl;
    cout << "  Volume fraction fibre : " << V_fiber*100 << " %" << endl;
    if (config.hasPores) {
        cout << "  Volume fraction pores : " << V_pore*100 << " %" << endl;
    }

    // On cherche maintenant les autres composantes du tenseur de rigidité du composite

    solver.clearBCs();
    solver.clearSystem();
    // Test de traction selon y
    // Encastrement en bas, force en haut
    for (int id : mesh.bottomNodes) {
        solver.setDirichletBC(id, 0, 0.0);
        solver.setDirichletBC(id, 1, 0.0);
    }
    for (int id : mesh.findNodesAtX(mesh.xMax / 2.0)) {
        solver.setDirichletBC(id, 1, 0.0);
    }
    // Force repartie en haut
    applyDistributedForce(solver, mesh, mesh.topNodes, totalForce, 1);
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.assemble();  // Re-assembler pour calculer les contraintes à partir des déplacements
    // Calcul de E2 et v21
    U = solver.getU();
    double uy_y = calcDisp(mesh.topNodes, 1);
    double epsilon_y_y = uy_y / H;
    double sigma_y = totalForce / A;
    comp.E2 = sigma_y / epsilon_y_y;
    comp.v12 = -calcDisp(mesh.rightNodes, 0) / uy_y;
    
    cout << "\n=== Propriétés effectives du composite (suite) ===" << endl;
    cout << "  E_2 : " << comp.E2/1e9 << " GPa" << endl;
    cout << "  v_12 : " << comp.v12 << endl;

    // Test de cisaillement
    solver.clearBCs();

    // Encastrement à gauche
    for (int id : mesh.leftNodes) {
        solver.setDirichletBC(id, 0, 0.0);
        solver.setDirichletBC(id, 1, 0.0);
    }

    // Force de cisaillement répartie en haut
    applyDistributedForce(solver, mesh, mesh.topNodes, totalForce, 0);
    solver.applyBC();

    solver.solveConjugateGradient();
    U = solver.getU();
    double ux_shear = calcDisp(mesh.topNodes, 0);
    double gamma12 = ux_shear / H;
    comp.G12 = sigma_y / gamma12;
    cout << "\n=== Propriétés effectives du composite (suite) ===" << endl;
    cout << "  G_12 : " << comp.G12/1e9 << " GPa" << endl;


}

