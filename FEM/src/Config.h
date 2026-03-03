#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <map>

class Config {
public:
    // Type de test
    std::string testType;  // "traction" ou "flexion"
    
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
    
    // Forces appliquées
    double forceValue;  // Valeur de la force (N)
    
    // Facteur d'échelle des dimensions
    double dimensionScale;  // Facteur pour convertir coordonnées maillage vers mètres (ex: 0.07e-6 pour 0.07 µm/pixel)
    
    // Fichiers de sortie
    std::string outputDir;
    std::string outputFilePrefix;
    
    Config();
    void loadFromFile(const std::string& filename);
    void print() const;

private:
    std::map<std::string, std::string> params;
    void parseFile(const std::string& filename);
    double getDouble(const std::string& key, double defaultValue) const;
    std::string getString(const std::string& key, const std::string& defaultValue) const;
};

#endif
