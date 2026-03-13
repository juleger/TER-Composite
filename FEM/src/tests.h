#ifndef TESTS_H
#define TESTS_H

#include "config.h"
#include <string>

// Fonctions de test pour différents cas de charge
void runTractionTest(const std::string& meshFile, const Config& config);
void runFlexionTest(const std::string& meshFile, const Config& config);
void runShearTest(const std::string& meshFile, const Config& config);
void runCompositeTest(const std::string& meshFile, const Config& config);
#endif
