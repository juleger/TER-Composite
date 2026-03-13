#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include <functional>
#include "mesh.h"
#include "material.h"

using ExactFn = std::function<Eigen::Vector2d(double, double)>;

class Solver {
    // Résout KU=F par gradient conjugué, assemble K depuis les Ke élémentaires,
    // et applique les conditions aux limites.

public:
    Solver(Mesh& mesh, double tolerance = 1e-4, int maxIterations = 500);

    void assemble();
    void applyBC();
    void solveConjugateGradient();
    void computeStrainStress();
    void Reinitialize();

    void setDirichletBC(int nodeId, int dof, double value);
    void setNeumannBC(int nodeId, int dof, double value);
    void clearNeumannBCs()  { _neumannBCs.clear(); }
    void clearDirichletBCs() { _dirichletBCs.clear(); }
    void clearBCs()   { clearDirichletBCs(); clearNeumannBCs(); }
    void clearSystem() { _K.setZero(); _F.setZero(); _U.setZero(); }

    Eigen::VectorXd getU()      const { return _U; }
    Eigen::MatrixXd getStress() const { return _stress; }
    Eigen::MatrixXd getStrain() const { return _strain; }
    double U(int nodeId, int dof) const { return _U(2*(nodeId-1) + dof); }
    double F(int nodeId, int dof) const { return _F(2*(nodeId-1) + dof); }

    void computeL2Error(ExactFn exact);

    void saveResults(const std::string& filename) const;
    void saveVTK(const std::string& filename);

    double errL2    = 0.0;
    double errL2_rel = 0.0;

private:
    Mesh& _mesh;
    Eigen::VectorXd _U;
    Eigen::SparseMatrix<double> _K;
    Eigen::VectorXd _F;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper,
        Eigen::IncompleteCholesky<double>> _cgSolver;
    Eigen::MatrixXd _stress;
    Eigen::MatrixXd _strain;
    double _tol;
    int _maxIter;
    std::map<int, double> _dirichletBCs;
    std::map<int, double> _neumannBCs;
};

#endif
