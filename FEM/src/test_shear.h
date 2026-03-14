#ifndef TEST_SHEAR_H
#define TEST_SHEAR_H

#include "tests_utils.h"
#include <string>
#include <vector>

void runShearTest(const std::string& meshFile, const Config& config);
void runShearTest(const std::vector<std::string>& meshFiles, const std::vector<double>& meshLc, const Config& config);

#endif
