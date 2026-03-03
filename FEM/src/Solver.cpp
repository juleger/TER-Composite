#include "Solver.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace Eigen;

Solver::Solver(Mesh& mesh, double tolerance, int maxIterations)
    : _mesh(mesh), _tol(tolerance), _maxIter(maxIterations) {
    
    int nbDofs = 2 * _mesh.nbNodes();
    
    _U.resize(nbDofs);
    _U.setZero();
    
    _F.resize(nbDofs);
    _F.setZero();
    
    _K.resize(nbDofs, nbDofs);
}

void Solver::setDirichletBC(int nodeId, int dof, double value) {
    int globalDof = 2 * (nodeId - 1) + dof;
    _dirichletBCs[globalDof] = value;
}

void Solver::setNeumannBC(int nodeId, int dof, double value) {
    int globalDof = 2 * (nodeId - 1) + dof;
    _neumannBCs[globalDof] = value;
}

void Solver::clearBCs() {
    _dirichletBCs.clear();
    _neumannBCs.clear();
}

void Solver::assemble() {
    cout << "Assemblage en cours..." << endl;
    
    vector<Triplet<double>> triplets;
    int nbElements = _mesh.elements.size();
    triplets.reserve(nbElements * 36);
    int count = 0;
    
    for (const auto& elem : _mesh.elements) {
        if (elem.Ke.norm() < 1e-20) continue;
        
        vector<int> dofMap;
        for (int node : elem.nodeIds) {
            dofMap.push_back(2*(node-1));
            dofMap.push_back(2*(node-1)+1);
        }
        
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                triplets.push_back(Triplet<double>(dofMap[i], dofMap[j], elem.Ke(i,j)));
            }
        }
        
        count++;
    }
    _K.setFromTriplets(triplets.begin(), triplets.end());
    cout << "Assemblage terminé : Matrice " << _K.rows() << "x" << _K.cols() 
         << ", nnz = " << _K.nonZeros() << endl;
}

void Solver::applyBC() {
    cout << "Application des conditions aux limites..." << endl;
    
    // Appliquer les forces (Neumann BC)
    for (const auto& force : _neumannBCs) {
        int dof = force.first;
        double value = force.second;
        _F(dof) += value;
    }
    // Méthode efficace : ne modifier que les coefficients existants
    for (const auto& disp : _dirichletBCs) {
        int dof = disp.first;
        double value = disp.second;
        
        // Mettre à zéro la ligne et la colonne correspondantes
        // en parcourant seulement les coefficients non-nuls
        for (int k = 0; k < _K.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(_K, k); it; ++it) {
                if (it.row() == dof || it.col() == dof) {
                    if (it.row() == dof && it.col() == dof) {
                        it.valueRef() = 1.0;
                    } else {
                        it.valueRef() = 0.0;
                    }
                }
            }
        }
        
        _F(dof) = value;
    }
    
    cout << "CL : " << _dirichletBCs.size() << " déplacements imposés, " 
         << _neumannBCs.size() << " forces appliquées" << endl;
}

void Solver::solveConjugateGradient() {
    cout << "\n Résolution par gradient conjugué..." << endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    ConjugateGradient<SparseMatrix<double>, Lower|Upper, DiagonalPreconditioner<double>> solver;
    solver.setTolerance(_tol);
    solver.setMaxIterations(_maxIter);
    solver.compute(_K);
    
    if (solver.info() != Success) {
        cerr << "Erreur : échec de l'initialisation du gradient conjugué" << endl;
        return;
    }
    _U = solver.solve(_F);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;

    if (solver.info() != Success) {
        cerr << "Erreur : le gradient conjugué n'a pas convergé" << endl;
        cerr << "Itérations: " << solver.iterations() << ", erreur: " << solver.error() << endl;
        cerr << "Temps de résolution: " << elapsed.count() << " s" << endl;
        return;
    }

    // Afficher résumé
    cout << "\n=== Résolution terminée ===" << endl;
    cout << "Itérations: " << solver.iterations() << endl;
    cout << "Temps: " << elapsed.count() << " s" << endl;
}

void Solver::saveResults(const string& filename) const {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Erreur : impossible d'ouvrir " << filename << endl;
        return;
    }
    
    file << "# Résultats de la simulation FEM\n";
    file << "# NodeID X Y Ux Uy Unorm\n";
    
    for (const auto& node : _mesh.nodes) {
        double ux = _U(2*(node.id-1));
        double uy = _U(2*(node.id-1)+1);
        double unorm = sqrt(ux*ux + uy*uy);
        file << node.id << " " << node.coords.x() << " " << node.coords.y() 
             << " " << ux << " " << uy << " " << unorm << "\n";
    }
    
    file.close();
}

void Solver::saveVTK(const string& filename) const {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Erreur : impossible d'ouvrir " << filename << endl;
        return;
    }
    
    // En-tête VTK
    file << "# vtk DataFile Version 3.0\n";
    file << "FEM Results\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n\n";
    
    // Points
    file << "POINTS " << _mesh.nbNodes() << " float\n";
    for (const auto& node : _mesh.nodes) {
        file << node.coords.x() << " " << node.coords.y() << " 0.0\n";
    }
    file << "\n";
    
    // Cellules (triangles)
    file << "CELLS " << _mesh.nbElements() << " " << (4 * _mesh.nbElements()) << "\n";
    for (const auto& elem : _mesh.elements) {
        file << "3 " << (elem.nodeIds[0]-1) << " " << (elem.nodeIds[1]-1) << " " << (elem.nodeIds[2]-1) << "\n";
    }
    file << "\n";
    
    // Types de cellules (5 = triangle)
    file << "CELL_TYPES " << _mesh.nbElements() << "\n";
    for (int i = 0; i < _mesh.nbElements(); i++) {
        file << "5\n";
    }
    file << "\n";
    
    // Données aux noeuds
    file << "POINT_DATA " << _mesh.nbNodes() << "\n";
    
    // Vecteur déplacement
    file << "VECTORS U float\n";
    for (const auto& node : _mesh.nodes) {
        file << _U(2*(node.id-1)) << " " << _U(2*(node.id-1)+1) << " 0.0\n";
    }
    
    file.close();
    cout << "Fichier VTK sauvegardé: " << filename << endl;
}
