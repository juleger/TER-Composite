#include "config.h"
#include "solver.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cctype>

using namespace std;

namespace {
PreconditionerType parsePreconditioner(const string& raw) {
    string key = raw;
    transform(key.begin(), key.end(), key.begin(), [](unsigned char c){ return static_cast<char>(tolower(c)); });
    if (key == "diag" || key == "diagonal") return PreconditionerType::Diagonal;
    if (key == "identity" || key == "none") return PreconditionerType::Identity;
    return PreconditionerType::IncompleteCholesky;
}
}

Config::Config() {
    // Valeurs par défaut
    testType = "traction";
    E = 200.0e9;
    nu = 0.3;
    rho = 7850.0;
    forceValue = 1000.0;
    outputVtk = "";
    preconditioner = "ic";
    solverTolerance = 1e-8;
    solverMaxIter = 2000;
    meshBaseDir = "";
    meshPrefix = "";
    meshSuffix = ".msh";
}

void Config::loadFromFile(const string& filename) {
    parseFile(filename);

    auto hasAny = [this](initializer_list<const char*> keys) {
        for (const char* k : keys) {
            if (params.find(k) != params.end()) return true;
        }
        return false;
    };
    auto getDoubleAny = [this](initializer_list<const char*> keys, double def) {
        for (const char* k : keys) {
            auto it = params.find(k);
            if (it != params.end()) {
                try {
                    return stod(it->second);
                } catch (...) {
                    cerr << "Erreur de conversion pour " << k << endl;
                }
            }
        }
        return def;
    };
    auto getStringAny = [this](initializer_list<const char*> keys, const string& def) {
        for (const char* k : keys) {
            auto it = params.find(k);
            if (it != params.end()) return it->second;
        }
        return def;
    };
    
    // Charger les paramètres
    testType = getStringAny({"test_type", "test"}, "traction");
    meshFile = getStringAny({"mesh_file", "mesh"}, "mesh/rectangle1.msh");

    std::string et = getStringAny({"element_type", "elem"}, "P1");
    if      (et == "P2") elementType = ElementType::P2;
    else if (et == "Q1") elementType = ElementType::Q1;
    else                 elementType = ElementType::P1;
    
    // Matériau 1 (matrice)
    E = getDoubleAny({"E", "Young_modulus"}, 200.0e9);
    nu = getDoubleAny({"nu", "Poisson_ratio"}, 0.3);
    rho = getDoubleAny({"rho", "density"}, 7850.0);
    
    // Matériau 2 (fibre) - optionnel
    hasFiber = hasAny({"E_fiber", "Young_modulus_fiber", "nu_fiber", "Poisson_ratio_fiber", "rho_fiber", "density_fiber"});
    E_fiber = getDoubleAny({"E_fiber", "Young_modulus_fiber"}, E);
    nu_fiber = getDoubleAny({"nu_fiber", "Poisson_ratio_fiber"}, nu);
    rho_fiber = getDoubleAny({"rho_fiber", "density_fiber"}, rho);
    
    // Matériau 3 (porosités) - optionnel
    hasPores = hasAny({"E_pore", "Young_modulus_pore", "nu_pore", "Poisson_ratio_pore", "rho_pore", "density_pore"});
    E_pore = getDoubleAny({"E_pore", "Young_modulus_pore"}, E);
    nu_pore = getDoubleAny({"nu_pore", "Poisson_ratio_pore"}, nu);
    rho_pore = getDoubleAny({"rho_pore", "density_pore"}, rho);
    
    forceValue = getDoubleAny({"F", "force_value"}, 1000.0);
    preconditioner = getStringAny({"precond", "preconditioner"}, "ic");
    solverTolerance = getDoubleAny({"tol", "solver_tol", "tolerance"}, 1e-8);
    solverMaxIter = static_cast<int>(getDoubleAny({"maxIter", "max_iter", "solver_max_iter"}, 2000.0));

    meshBaseDir = getStringAny({"mesh_base_dir", "conv_mesh_base_dir"}, "");
    meshPrefix = getStringAny({"mesh_prefix", "conv_mesh_prefix"}, "");
    meshSuffix = getStringAny({"mesh_suffix", "conv_mesh_suffix"}, ".msh");
    if (!meshSuffix.empty() && meshSuffix[0] != '.') meshSuffix = "." + meshSuffix;

    const string lcRaw = getStringAny({"mesh_lc", "mesh_lc_list", "conv_mesh_lc"}, "");
    meshLcTokens = splitList(lcRaw);

    outputVtk = getStringAny({"output_vtk", "vtk", "output"}, "");
    if (outputVtk.empty()) {
        // Compatibilité anciens fichiers de config
        string legacyPrefix = getString("output_prefix", "");
        if (!legacyPrefix.empty()) outputVtk = legacyPrefix + ".vtk";
    }
}

vector<string> Config::splitList(const string& raw) const {
    vector<string> items;
    string token;
    for (char c : raw) {
        if (c == ',' || c == ';' || isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) {
                items.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) items.push_back(token);
    return items;
}

void Config::buildConvergenceMeshes(vector<string>& meshFiles, vector<double>& meshLc) const {
    vector<string> lcTokens = meshLcTokens;
    string baseDir = meshBaseDir;
    string prefix = meshPrefix;
    string suffix = meshSuffix.empty() ? ".msh" : meshSuffix;

    if (lcTokens.empty()) {
        cout << "Aucun mesh_lc spécifié, pas de série de maillages pour convergence." << endl;
        return;
    }
    meshFiles.clear();
    meshLc.clear();
    meshFiles.reserve(lcTokens.size());
    meshLc.reserve(lcTokens.size());

    for (const auto& tok : lcTokens) {
        try {
            meshLc.push_back(stod(tok));
        } catch (...) {
            cerr << "Valeur de mesh_lc invalide: " << tok << endl;
            continue;
        }
        meshFiles.push_back(baseDir + "/" + prefix + tok + suffix);
    }
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

string Config::vtkPath(const string& meshFile) const {
    namespace fs = std::filesystem;

    string name = outputVtk;
    if (name.empty()) {
        name = fs::path(meshFile).stem().string() + ".vtk";
    } else {
        name = fs::path(name).filename().string();
        if (fs::path(name).extension() != ".vtk")
            name += ".vtk";
    }
    return string("results/") + name;
}

void Config::print() const {
    const char* etNames[] = {"P1", "P2", "Q1"};
    cout << "----------------- Configuration ----------------" << endl;
    cout << "Type de test: " << testType << endl;
    cout << "Type d'élément: " << etNames[(int)elementType] << endl;
    cout << "Fichier de maillage: " << meshFile << endl;
    cout << "Matériau 1 (Matrice) - E: " << E << " Pa, nu: " << nu << ", rho: " << rho << " kg/m3" << endl;
    if (hasFiber) {
        cout << "Matériau 2 (Fibre) - E: " << E_fiber << " Pa, nu: " << nu_fiber << ", rho: " << rho_fiber << " kg/m3" << endl;
    }
    if (hasPores) {
        cout << "Matériau 3 (Pore) - E: " << E_pore << " Pa, nu: " << nu_pore << ", rho: " << rho_pore << " kg/m3" << endl;
    }
    cout << "Force appliquée: " << forceValue << " N" << endl;
    cout << "Solveur - precond: " << preconditioner << ", tol: " << solverTolerance << ", maxIter: " << solverMaxIter << "\n" << endl; 
}

void applySolverConfig(Solver& solver, const Config& config) {
    solver.setTolerance(config.solverTolerance);
    solver.setMaxIterations(config.solverMaxIter);
    solver.setPreconditioner(parsePreconditioner(config.preconditioner));
}
