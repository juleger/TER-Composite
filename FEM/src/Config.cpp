#include "Config.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

Config::Config() {
    // Valeurs par défaut
    testType = "traction";
    E = 200.0e9;
    nu = 0.3;
    rho = 7850.0;
    forceValue = 1000.0;
    dimensionScale = 1.0;  // Par défaut, pas de scaling
    outputDir = "results";
    outputFilePrefix = "test";
}

void Config::loadFromFile(const string& filename) {
    parseFile(filename);
    
    // Charger les paramètres
    testType = getString("test_type", "traction");
    meshFile = getString("mesh_file", "mesh/rectangle1.msh");
    
    // Matériau 1 (matrice)
    E = getDouble("Young_modulus", 200.0e9);
    nu = getDouble("Poisson_ratio", 0.3);
    rho = getDouble("density", 7850.0);
    
    // Matériau 2 (fibre) - optionnel
    E_fiber = getDouble("Young_modulus_fiber", E);
    nu_fiber = getDouble("Poisson_ratio_fiber", nu);
    rho_fiber = getDouble("density_fiber", rho);
    hasFiber = (params.find("Young_modulus_fiber") != params.end());
    
    forceValue = getDouble("force_value", 1000.0);
    dimensionScale = getDouble("dimension_scale", 1.0);
    outputDir = getString("output_dir", "results");
    outputFilePrefix = getString("output_prefix", "test");
}

void Config::parseFile(const string& filename) {
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Attention : impossible d'ouvrir " << filename << endl;
        cerr << "Utilisation des valeurs par défaut" << endl;
        return;
    }
    
    string line;
    while (getline(file, line)) {
        // Ignorer les commentaires et lignes vides
        if (line.empty() || line[0] == '#') continue;
        
        // Parser la ligne "key = value"
        size_t pos = line.find('=');
        if (pos != string::npos) {
            string key = line.substr(0, pos);
            string value = line.substr(pos + 1);
            
            // Enlever les espaces
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            params[key] = value;
        }
    }
    
    file.close();
}

double Config::getDouble(const string& key, double defaultValue) const {
    auto it = params.find(key);
    if (it != params.end()) {
        try {
            return stod(it->second);
        } catch (...) {
            cerr << "Erreur de conversion pour " << key << endl;
        }
    }
    return defaultValue;
}

string Config::getString(const string& key, const string& defaultValue) const {
    auto it = params.find(key);
    if (it != params.end()) {
        return it->second;
    }
    return defaultValue;
}

void Config::print() const {
    cout << "=== Configuration ===" << endl;
    cout << "Type de test: " << testType << endl;
    cout << "Fichier de maillage: " << meshFile << endl;
    cout << "\nMatériau 1 (matrice):" << endl;
    cout << " E = " << E << " Pa," << " nu = " << nu << ", rho = " << rho << " kg/m³" << endl;
    
    if (hasFiber) {
        cout << "\nMatériau 2 (fibre):" << endl;
        cout << "E = " << E_fiber << " Pa," << " nu = " << nu_fiber << ", rho = " << rho_fiber << " kg/m³" << endl;
    }
    
    cout << "\nForce appliquée: " << forceValue << " N" << endl;    cout << "Facteur d'échelle des dimensions: " << dimensionScale << " m/unité" << endl;    cout << endl;
}
