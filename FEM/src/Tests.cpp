#include "Tests.h"
#include "Mesh.h"
#include "Material.h"
#include "Solver.h"
#include "MeshReader.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

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
    mesh.scaleCoordinates(config.dimensionScale);  // Appliquer le facteur d'échelle
    
    cout << "Noeuds: " << mesh.nbNodes() << ", Eléments: " << mesh.nbElements() << endl;
    cout << "Dimensions: " << mesh.width() << " x " << mesh.height() << " m\n" << endl;
    
    // Résolution
    Solver solver(mesh);
    solver.assemble();
    
    // CL: encastrement à gauche, force à droite
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0);
    for (int id : mesh.findNodesAtY(mesh.yMax / 2.0)) solver.setDirichletBC(id, 1, 0.0);
    
    // Force répartie à droite
    vector<pair<int, double>> rightNodesY;
    for (int id : mesh.rightNodes) {
        rightNodesY.push_back({id, mesh.getNode(id).coords.y()});
    }
    sort(rightNodesY.begin(), rightNodesY.end(), 
         [](const pair<int,double>& a, const pair<int,double>& b) { return a.second < b.second; });
    
    double totalForce = config.forceValue;
    for (size_t i = 0; i < rightNodesY.size(); i++) {
        double len;
        if (i == 0) {
            len = (rightNodesY[1].second - rightNodesY[0].second) / 2.0;
        } else if (i == rightNodesY.size() - 1) {
            len = (rightNodesY[i].second - rightNodesY[i-1].second) / 2.0;
        } else {
            len = (rightNodesY[i+1].second - rightNodesY[i-1].second) / 2.0;
        }
        solver.setNeumannBC(rightNodesY[i].first, 0, totalForce * len / mesh.height());
    }
    
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.saveResults(config.outputDir + "/displacement_" + config.outputFilePrefix + ".txt");
    solver.saveVTK(config.outputDir + "/results_" + config.outputFilePrefix + ".vtk");
    
    // Validation avec résultats théoriques
    Eigen::VectorXd U = solver.getU();
    auto calcDisp = [&](const vector<int>& nodes, int dof) {
        double sum = 0;
        for (int id : nodes) sum += U(2*(id-1) + dof);
        return sum / nodes.size();
    };
    
    double ux = calcDisp(mesh.rightNodes, 0);
    double uy = (abs(calcDisp(mesh.topNodes, 1)) + abs(calcDisp(mesh.bottomNodes, 1))) / 2.0;
    
    double A = mesh.height(), L = mesh.width();
    // Delta L = FL/AE
    double ux_theo = totalForce * L / (A * config.E);
    double uy_theo = config.nu * totalForce * mesh.height() / (2.0 * A * config.E);
    
    cout << "\n=== Résultats ===" << endl;
    cout << "Allongement x: " << ux << " m (théo: " << ux_theo << ", erreur: " 
         << abs(ux-ux_theo)/ux_theo*100 << "%)" << endl;
    cout << "Contraction y: " << uy << " m (théo: " << uy_theo << ", erreur: " 
         << abs(uy-uy_theo)/uy_theo*100 << "%)" << endl;
}

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
    mesh.scaleCoordinates(config.dimensionScale);  // Appliquer le facteur d'échelle
    
    cout << "Noeuds: " << mesh.nbNodes() << ", Eléments: " << mesh.nbElements() << endl;
    cout << "Dimensions: " << mesh.width() << " x " << mesh.height() << " m\n" << endl;
    
    Solver solver(mesh, 1e-6, 2000);
    solver.assemble();
    
    // Encastrement complet à gauche
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
    solver.saveResults(config.outputDir + "/displacement_" + config.outputFilePrefix + ".txt");
    solver.saveVTK(config.outputDir + "/results_" + config.outputFilePrefix + ".vtk");
    
    // Flèche au point d'application de la force
    Eigen::VectorXd U = solver.getU();
    double fleche = abs(U(2*(nodeForce-1) + 1));
    
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
}

void runCompositeTest(const string& meshFile, const Config& config) {
    cout << "=== Test Composite (matrice + fibre) ===" << endl;
    cout << "Maillage: " << meshFile << endl;
    
    // Créer les deux matériaux
    Material matrix(config.E, config.nu, config.rho);
    Material fiber(config.E_fiber, config.nu_fiber, config.rho_fiber);
    
    // Charger le maillage
    Mesh mesh;
    MeshReader reader(&mesh);
    reader.setMaterial(1, &matrix);  // Matériau 1 = matrice
    reader.setMaterial(2, &fiber);   // Matériau 2 = fibre
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();
    mesh.scaleCoordinates(config.dimensionScale);  // Appliquer le facteur d'échelle
    
    cout << "Noeuds: " << mesh.nbNodes() << ", Eléments: " << mesh.nbElements() << endl;
    cout << "Dimensions: " << mesh.width() << " x " << mesh.height() << " m\n" << endl;
    
    // Résolution
    Solver solver(mesh, 1e-6, 2000);
    solver.assemble();
    
    // Conditions aux limites: encastrement à gauche, force à droite
    for (int id : mesh.leftNodes) solver.setDirichletBC(id, 0, 0.0);
    for (int id : mesh.findNodesAtY(mesh.yMax / 2.0)) solver.setDirichletBC(id, 1, 0.0);
    
    // Force répartie à droite
    vector<pair<int, double>> rightNodesY;
    for (int id : mesh.rightNodes) {
        rightNodesY.push_back({id, mesh.getNode(id).coords.y()});
    }
    sort(rightNodesY.begin(), rightNodesY.end(), 
         [](const pair<int,double>& a, const pair<int,double>& b) { return a.second < b.second; });
    
    double totalForce = config.forceValue;
    for (size_t i = 0; i < rightNodesY.size(); i++) {
        double len;
        if (i == 0) {
            len = (rightNodesY[1].second - rightNodesY[0].second) / 2.0;
        } else if (i == rightNodesY.size() - 1) {
            len = (rightNodesY[i].second - rightNodesY[i-1].second) / 2.0;
        } else {
            len = (rightNodesY[i+1].second - rightNodesY[i-1].second) / 2.0;
        }
        solver.setNeumannBC(rightNodesY[i].first, 0, totalForce * len / mesh.height());
    }
    
    solver.applyBC();
    solver.solveConjugateGradient();
    solver.saveResults(config.outputDir + "/displacement_" + config.outputFilePrefix + ".txt");
    solver.saveVTK(config.outputDir + "/results_" + config.outputFilePrefix + ".vtk");
    
    // Résultats
    Eigen::VectorXd U = solver.getU();
    auto calcDisp = [&](const vector<int>& nodes, int dof) {
        double sum = 0;
        for (int id : nodes) sum += U(2*(id-1) + dof);
        return sum / nodes.size();
    };
    
    double ux = calcDisp(mesh.rightNodes, 0);
    double uy = (abs(calcDisp(mesh.topNodes, 1)) + abs(calcDisp(mesh.bottomNodes, 1))) / 2.0;
    
    // Calcul des propriétés effectives
    double L = mesh.width();
    double H = mesh.height();
    double A = H;  // épaisseur unité
    
    // Déformations moyennes
    double epsilon_x = ux / L;
    double epsilon_y = -uy / (H/2.0);
    
    // Contrainte appliquée
    double sigma_x = totalForce / A; 
    
    // Propriétés effectives homogénéisées
    double E_eff = sigma_x / epsilon_x;            // Module de Young effectif
    double nu_eff = -epsilon_y / epsilon_x;        // Coefficient de Poisson effectif
    
    cout << "\n=== Résultats ===" << endl;
    cout << "Déplacements :" << endl;
    cout << "  Allongement moyen x : " << ux << " m (" << epsilon_x*100 << "%)" << endl;
    cout << "  Contraction moyenne y : " << uy << " m (" << epsilon_y*100 << "%)" << endl;
    
    cout << "\n=== Propriétés effectives du composite ===" << endl;
    cout << "  Module de Young effectif (E_eff) : " << E_eff/1e9 << " GPa" << endl;
    cout << "  Coefficient de Poisson effectif (ν_eff) : " << nu_eff << endl;
    
    // Comparaison avec la matrice pure
    cout << "\nComparaison avec matrice pure :" << endl;
    cout << "  E_eff/E_matrix: " << E_eff/config.E << " (rigidification)" << endl;
    cout << "  ν_eff - ν_matrix: " << (nu_eff - config.nu) << endl;

}

