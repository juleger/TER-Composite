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

enum class PreconditionerType {
    IncompleteCholesky,
    Diagonal,
    Identity
};

class Solver {
    // Résout KU=F par gradient conjugué, assemble K depuis les Ke élémentaires,
    // et applique les conditions aux limites.

public:
    Solver(Mesh& mesh, double tolerance = 1e-6, int maxIterations = 1000);

    void assemble();
    void applyBC();
    void solveConjugateGradient();
    void computeStrainStress();
    void Reinitialize();

    void setDirichletBC(int nodeId, int dof, double value);
    void setNeumannBC(int nodeId, int dof, double value);
    void setTolerance(double tol) { _tol = tol; }
    void setMaxIterations(int maxIter) { _maxIter = maxIter; }
    void setPreconditioner(PreconditionerType p) { _preconditioner = p; }
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
    double computeInternalEnergy();
    double computeExternalWork() const;
    int getLastSolveIterations() const { return _lastSolveIterations; }
    double getLastSolveError() const { return _lastSolveError; }
    double getLastSolveTimeSec() const { return _lastSolveTimeSec; }

    void saveResults(const std::string& filename) const;
    void saveVTK(const std::string& filename);

    double errL2    = 0.0;
    double errL2_rel = 0.0;

private:
    Mesh& _mesh;
    Eigen::VectorXd _U;
    Eigen::SparseMatrix<double> _K;
    Eigen::VectorXd _F;
    Eigen::MatrixXd _stress;
    Eigen::MatrixXd _strain;
    double _tol;
    int _maxIter;
    PreconditionerType _preconditioner = PreconditionerType::IncompleteCholesky;
    int _lastSolveIterations = 0;
    double _lastSolveError = 0.0;
    double _lastSolveTimeSec = 0.0;
    std::map<int, double> _dirichletBCs;
    std::map<int, double> _neumannBCs;
};

#endif
