#include "Config.h"
#include "Tests.h"
#include <iostream>

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
    if (config.testType == "flexion") {
        runFlexionTest(config.meshFile, config);
    } else if (config.testType == "composite") {
        runCompositeTest(config.meshFile, config);
    } else {
        runTractionTest(config.meshFile, config);
    }
    
    return 0;
}
