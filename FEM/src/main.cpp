#include "config.h"
#include "tests.h"
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.txt>" << endl;
        cerr << "  Exemple: " << argv[0] << " config/traction_config.txt" << endl;
        return 1;
    }
    
    string configFile = argv[1];
    
    // Charger la config
    Config config;
    config.loadFromFile(configFile);
    config.print();
    
    // Exécuter le test approprié
    if (config.testType == "composite") {
        runCompositeTest(config.meshFile, config);
    } else if (config.testType == "flexion") {
        vector<double> meshLc;
        vector<string> meshFiles;
        config.buildConvergenceMeshes(meshFiles, meshLc);
        runFlexionTest(meshFiles, meshLc, config);
    } else if (config.testType == "shear") {
        vector<double> meshLc;
        vector<string> meshFiles;
        config.buildConvergenceMeshes(meshFiles, meshLc);
        runShearTest(meshFiles, meshLc, config);
    } else {
        vector<double> meshLc;
        vector<string> meshFiles;
        config.buildConvergenceMeshes(meshFiles, meshLc);
        runTractionTest(meshFiles, meshLc, config);
    }
    
    return 0;
}
