#ifndef TESTS_UTILS_H
#define TESTS_UTILS_H

#include "config.h"
#include "mesh.h"
#include "solver.h"
#include "material.h"
#include <string>
#include <vector>
#include <functional>

struct ConvergenceResult {
	double h;
	int    nElems;
	double L2rel;
	double deltaWrel;
	double errErel = -1.0;
	double errNurel = -1.0;
	double errGrel = -1.0;
};

struct ReportOptions {
    bool showL2 = true;
    bool showDeltaW = true;
    bool showErrE = false;
    bool showErrNu = false;
    bool showErrG = false;
};

void applyDistributedForce(Solver& solver, const Mesh& mesh, const std::vector<int>& nodeIds,
                           std::function<double(double)> profile, int dof);
void applyDistributedForce(Solver& solver, const Mesh& mesh, const std::vector<int>& nodeIds,
                           double totalForce, int dof);

void loadAndInitMesh(const std::string& meshFile, Material& material, Mesh& mesh);

void printConvergenceTable(const std::vector<ConvergenceResult>& results, const std::string& label,
                           const ReportOptions& opt,
                           const std::function<double(const ConvergenceResult&)>& orderMetric);

void exportConvergenceCSV(const std::vector<ConvergenceResult>& results, const std::string& path,
                          const ReportOptions& opt,
                          const std::function<double(const ConvergenceResult&)>& orderMetric);

std::string convergenceVtkPath(const Config& config, double lc);

void applyShearDirichletBC(Solver& solver, const Mesh& mesh, double gammaTarget);

std::vector<ConvergenceResult> runConvergence(
    const std::vector<std::string>& meshFiles,
    const std::vector<double>& meshLc,
    const std::function<bool(size_t, const std::string&, double, ConvergenceResult&)>& runOne);

#endif
