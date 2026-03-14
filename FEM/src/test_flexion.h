#ifndef TEST_FLEXION_H
#define TEST_FLEXION_H

#include "tests_utils.h"
#include <string>
#include <vector>

void runFlexionTest(const std::vector<std::string>& meshFiles, const std::vector<double>& meshLc, const Config& config);
void runFlexionTest(const std::string& meshFile, const Config& config);

#endif
