#include "solver.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace Eigen;

Solver::Solver(Mesh& mesh, double tolerance, int maxIterations)
    : _mesh(mesh), _tol(tolerance), _maxIter(maxIterations) {
    
    int nbDofs = 2 * _mesh.nbNodes();
    
    _U.resize(nbDofs), _U.setZero();
    _F.resize(nbDofs), _F.setZero();
    _K.resize(nbDofs, nbDofs);
    Eigen::initParallel();
    cout << "Threads Eigen : " << Eigen::nbThreads() << endl;
}

void Solver::setDirichletBC(int nodeId, int dof, double value) {
    int globalDof = 2 * (nodeId - 1) + dof;
    _dirichletBCs[globalDof] = value;
}

void Solver::setNeumannBC(int nodeId, int dof, double value) {
    int globalDof = 2 * (nodeId - 1) + dof;
    _neumannBCs[globalDof] = value;
}

void Solver::assemble() {
    cout << "Assemblage en cours..." << endl;

    int nbElements = _mesh.elements.size();

    // Chaque thread accumule ses triplets localement → pas de verrou
    int nbThreads = omp_get_max_threads();
    vector<vector<Triplet<double>>> threadTriplets(nbThreads);
    for (auto& t : threadTriplets)
        t.reserve(nbElements * 144 / nbThreads + 1);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int e = 0; e < nbElements; e++) {
        const auto& elem = _mesh.elements[e];
        if (elem->Ke.norm() < 1e-20) continue;

        int tid = omp_get_thread_num();
        int n = elem->nDofs();
        vector<int> dofMap;
        dofMap.reserve(n);
        for (int nid : elem->nodeIds) {
            dofMap.push_back(2*(nid-1));
            dofMap.push_back(2*(nid-1)+1);
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                threadTriplets[tid].emplace_back(dofMap[i], dofMap[j], elem->Ke(i,j));
    }

    // Fusion séquentielle des triplets
    vector<Triplet<double>> triplets;
    triplets.reserve(nbElements * 144);
    for (auto& t : threadTriplets)
        triplets.insert(triplets.end(), t.begin(), t.end());

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

    // Eigen parallélise les opérations internes (SpMV, dot, axpy) via OpenMP
    Eigen::setNbThreads(omp_get_max_threads());

    _cgSolver.setTolerance(_tol);
    _cgSolver.setMaxIterations(_maxIter);
    _cgSolver.compute(_K);
    
    if (_cgSolver.info() != Success) {
        cerr << "Erreur : échec de l'initialisation du gradient conjugué" << endl;
        return;
    }
    _U = _cgSolver.solve(_F);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;

    if (_cgSolver.info() != Success) {
        cerr << "Erreur : le gradient conjugué n'a pas convergé" << endl;
        cerr << "Itérations: " << _cgSolver.iterations() << ", erreur: " << _cgSolver.error() << endl;
        cerr << "Temps de résolution: " << elapsed.count() << " s" << endl;
        return;
    }

    // Afficher résumé
    cout << "\n=== Résolution terminée ===" << endl;
    cout << "Itérations: " << _cgSolver.iterations() << endl;
    cout << "Temps: " << elapsed.count() << " s" << endl;
}

void Solver::computeStrainStress() {
    cout << "\nCalcul des contraintes et déformations..." << endl;
    int nbElements = _mesh.elements.size();
    _strain.resize(nbElements, 3);
    _stress.resize(nbElements, 3);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nbElements; i++) {
        const auto& elem = _mesh.elements[i];
        int n = elem->nNodes();
        VectorXd ue(2*n);
        for (int j = 0; j < n; j++) {
            int nid = elem->nodeIds[j];
            ue(2*j)   = U(nid, 0);
            ue(2*j+1) = U(nid, 1);
        }
        _strain.row(i) = elem->B * ue;
        _stress.row(i) = elem->material->C * _strain.row(i).transpose();
    }
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
        double ux = U(node.id, 0);
        double uy = U(node.id, 1);
        double unorm = sqrt(ux*ux + uy*uy);
        file << node.id << " " << node.coords.x() << " " << node.coords.y() 
             << " " << ux << " " << uy << " " << unorm << "\n";
    }
    
    file.close();
}

void Solver::saveVTK(const string& filename) {
    // Calcule contraintes/déformations si pas encore fait
    if (_stress.rows() != _mesh.nbElements())
        computeStrainStress();

    vector<Eigen::Vector3d> nodalStress(_mesh.nbNodes(), Eigen::Vector3d::Zero());
    vector<Eigen::Vector3d> nodalStrain(_mesh.nbNodes(), Eigen::Vector3d::Zero());
    vector<double> nodalWeight(_mesh.nbNodes(), 0.0);

    for (int e = 0; e < _mesh.nbElements(); ++e) {
        const auto& elem = _mesh.elements[e];
        Eigen::Vector3d sigma = _stress.row(e).transpose();
        Eigen::Vector3d eps   = _strain.row(e).transpose();
        for (int nid : elem->nodeIds) {
            int idx = nid - 1;
            nodalStress[idx] += elem->area * sigma;
            nodalStrain[idx] += elem->area * eps;
            nodalWeight[idx] += elem->area;
        }
    }

    for (int i = 0; i < _mesh.nbNodes(); ++i) {
        if (nodalWeight[i] > 0.0) {
            nodalStress[i] /= nodalWeight[i];
            nodalStrain[i] /= nodalWeight[i];
        }
    }

    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Erreur : impossible d'ouvrir " << filename << endl;
        return;
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "FEM Results\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n\n";

    // Points
    file << "POINTS " << _mesh.nbNodes() << " double\n";
    for (const auto& node : _mesh.nodes)
        file << node.coords.x() << " " << node.coords.y() << " 0.0\n";
    file << "\n";

    // Cellules
    int cellDataSize = 0;
    for (const auto& elem : _mesh.elements)
        cellDataSize += 1 + elem->nNodes();
    file << "CELLS " << _mesh.nbElements() << " " << cellDataSize << "\n";
    for (const auto& elem : _mesh.elements) {
        file << elem->nNodes();
        for (int nid : elem->nodeIds) file << " " << (nid - 1);
        file << "\n";
    }
    file << "\n";

    file << "CELL_TYPES " << _mesh.nbElements() << "\n";
    for (const auto& elem : _mesh.elements)
        file << elem->vtkType() << "\n";
    file << "\n";

    // ── POINT_DATA ──────────────────────────────────────────────
    file << "POINT_DATA " << _mesh.nbNodes() << "\n";

    file << "VECTORS U double\n";
    for (const auto& node : _mesh.nodes)
        file << U(node.id, 0) << " " << U(node.id, 1) << " 0.0\n";
    file << "\n";

    // Notation de Voigt explicite: [xx, yy, xy]
    file << "FIELD FieldData 2\n";

    file << "sigma_voigt 3 " << _mesh.nbNodes() << " double\n";
    for (int i = 0; i < _mesh.nbNodes(); ++i) {
        const double sxx = nodalStress[i](0);
        const double syy = nodalStress[i](1);
        const double sxy = nodalStress[i](2);
        file << sxx << " " << syy << " " << sxy << "\n";
    }

    file << "eps_voigt 3 " << _mesh.nbNodes() << " double\n";
    for (int i = 0; i < _mesh.nbNodes(); ++i) {
        const double exx = nodalStrain[i](0);
        const double eyy = nodalStrain[i](1);
        const double exy = nodalStrain[i](2);
        file << exx << " " << eyy << " " << exy << "\n";
    }
    file << "\n";

    file.close();
    cout << "Fichier VTK sauvegardé: " << filename << endl;
}

void Solver::Reinitialize() {
    int nbDofs = 2 * _mesh.nbNodes();
    _U.setZero();
    _F.setZero();
    _K.resize(nbDofs, nbDofs);
    _K.setZero();
    _dirichletBCs.clear();
    _neumannBCs.clear();
}

void Solver::computeL2Error(ExactFn exact) {
    double err2 = 0.0, ref2 = 0.0;
    for (const auto& elem : _mesh.elements) {
        Eigen::VectorXd N = elem->shapeAtCentroid();
        double cx = 0, cy = 0, ux = 0, uy = 0;
        for (int i = 0; i < elem->nNodes(); i++) {
            cx += N(i) * elem->nodes[i]->coords.x();
            cy += N(i) * elem->nodes[i]->coords.y();
            ux += N(i) * U(elem->nodeIds[i], 0);
            uy += N(i) * U(elem->nodeIds[i], 1);
        }

        Eigen::Vector2d uex = exact(cx, cy);
        double ex = ux - uex.x(), ey = uy - uex.y();
        err2 += (ex*ex + ey*ey) * elem->area;
        ref2 += (uex.x()*uex.x() + uex.y()*uex.y()) * elem->area;
    }
    errL2     = std::sqrt(err2);
    errL2_rel = (ref2 > 0.0) ? errL2 / std::sqrt(ref2) : 0.0;
}
