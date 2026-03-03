#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include "Mesh.h"
#include "Material.h"

class Solver {
    // Classe permettant de résoudre le système global KU=F avec la méthode du gradient conjugué.
    // Elle assemble également la matrice de rigidité globale K à partir des matrices élémentaires Ke (propre aux éléments)
    // Et applique les conditions aux limites sur le système.

    private:
        Mesh& _mesh;
        Eigen::VectorXd _U;
        Eigen::SparseMatrix<double> _K;
        Eigen::VectorXd _F;
        
        double _tol;
        int _maxIter;
        
        // Conditions aux limites
        std::map<int, double> _dirichletBCs; // globalDof -> prescribed displacement
        std::map<int, double> _neumannBCs; // globalDof -> applied force
        
    public:
        Solver(Mesh& mesh, double tolerance = 1e-4, int maxIterations = 500);

        void assemble();
        void applyBC();
        void solveConjugateGradient(); 
        
        // Méthodes pour définir les CL
        void setDirichletBC(int nodeId, int dof, double value);
        void setNeumannBC(int nodeId, int dof, double value);
        void clearBCs();
        
        Eigen::VectorXd getU() const { return _U; }
        void saveResults(const std::string& filename) const;
        void saveVTK(const std::string& filename) const;
};

#endif
