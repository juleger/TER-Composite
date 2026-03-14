#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <map>
#include <vector>

enum class ElementType { P1, P2, Q1 };
class Solver;

class Config {
public:
    // Type de test
    std::string testType;  // "traction" ou "flexion"

    // Type d'élément FEM
    ElementType elementType;

    // Fichier de maillage
    std::string meshFile;
    
    // Propriétés matériau 1 (matrice ou unique)
    double E;      // Module de Young (Pa)
    double nu;     // Coefficient de Poisson
    double rho;    // Densité (kg/m³)
    
    // Propriétés matériau 2 (fibre) - optionnel
    double E_fiber;    // Module de Young fibre (Pa)
    double nu_fiber;   // Coefficient de Poisson fibre
    double rho_fiber;  // Densité fibre (kg/m³)
    bool hasFiber;     // Indique si un second matériau est défini

    bool hasPores;    // Indique si le maillage contient des porosités (matériau = 3)
    double E_pore;
    double nu_pore;
    double rho_pore;


    double forceValue;  // Valeur de la force (N)

    // Paramètres solveur
    std::string preconditioner;  // ic, ilu, diag...
    double solverTolerance;
    int solverMaxIter;

    // Paramètres de série de maillages (convergence)
    std::string meshBaseDir;
    std::string meshPrefix;
    std::string meshSuffix;
    std::vector<std::string> meshLcTokens;
    
    // Fichier de sortie VTK (nom de fichier uniquement, dossier forcé à results/)
    std::string outputVtk;
    
    Config();
    void loadFromFile(const std::string& filename);
    void print() const;
    std::string vtkPath(const std::string& meshFile) const;
    void buildConvergenceMeshes(std::vector<std::string>& meshFiles,
                                std::vector<double>& meshLc) const;

private:
    std::map<std::string, std::string> params;
    void parseFile(const std::string& filename);
    double getDouble(const std::string& key, double defaultValue) const;
    std::string getString(const std::string& key, const std::string& defaultValue) const;
    std::vector<std::string> splitList(const std::string& raw) const;
};

void applySolverConfig(Solver& solver, const Config& config);

#endif
