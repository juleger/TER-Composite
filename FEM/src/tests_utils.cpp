#include "tests_utils.h"
#include "meshReader.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <functional>
#include <sstream>

using namespace std;

// Applique un profil quelconque de force nodale sur une face (intégration trapézoïdale)
void applyDistributedForce(Solver& solver, const Mesh& mesh, const vector<int>& nodeIds, function<double(double)> profile, int dof) {
    if (nodeIds.empty()) return;

    vector<pair<int, double>> nodesS;
    nodesS.reserve(nodeIds.size());

    double xMin = numeric_limits<double>::infinity();
    double xMax = -numeric_limits<double>::infinity();
    double yMin = numeric_limits<double>::infinity();
    double yMax = -numeric_limits<double>::infinity();
    for (int id : nodeIds) {
        const auto& p = mesh.getNode(id).coords;
        xMin = min(xMin, p.x());
        xMax = max(xMax, p.x());
        yMin = min(yMin, p.y());
        yMax = max(yMax, p.y());
    }

    const bool varyAlongX = ((xMax - xMin) >= (yMax - yMin));
    for (int id : nodeIds) {
        const auto& p = mesh.getNode(id).coords;
        nodesS.push_back({id, varyAlongX ? p.x() : p.y()});
    }

    sort(nodesS.begin(), nodesS.end(),
         [](const pair<int,double>& a, const pair<int,double>& b){ return a.second < b.second; });

    if (nodesS.size() == 1) {
        solver.setNeumannBC(nodesS[0].first, dof, profile(nodesS[0].second));
        return;
    }

    for (size_t i = 0; i < nodesS.size(); ++i) {
        double len;
        if (i == 0)
            len = (nodesS[1].second - nodesS[0].second) / 2.0;
        else if (i == nodesS.size() - 1)
            len = (nodesS[i].second - nodesS[i-1].second) / 2.0;
        else
            len = (nodesS[i+1].second - nodesS[i-1].second) / 2.0;
        solver.setNeumannBC(nodesS[i].first, dof, profile(nodesS[i].second) * len);
    }
}

// Profil uniforme: force totale répartie proportionnellement à la longueur
void applyDistributedForce(Solver& solver, const Mesh& mesh, const vector<int>& nodeIds, double totalForce, int dof) {
    if (nodeIds.empty()) return;

    double xMin = numeric_limits<double>::infinity();
    double xMax = -numeric_limits<double>::infinity();
    double yMin = numeric_limits<double>::infinity();
    double yMax = -numeric_limits<double>::infinity();
    for (int id : nodeIds) {
        const auto& p = mesh.getNode(id).coords;
        xMin = min(xMin, p.x());
        xMax = max(xMax, p.x());
        yMin = min(yMin, p.y());
        yMax = max(yMax, p.y());
    }

    const bool varyAlongX = ((xMax - xMin) >= (yMax - yMin));
    const double boundaryLength = max(varyAlongX ? (xMax - xMin) : (yMax - yMin), 1e-30);
    applyDistributedForce(solver, mesh, nodeIds,
        [totalForce, boundaryLength](double) { return totalForce / boundaryLength; }, dof);
}

// Charge un maillage depuis un fichier Gmsh et l'initialise (matériau 1)
void loadAndInitMesh(const string& meshFile, Material& material, Mesh& mesh) {
    cout << "\n" << string(80, '-') << "\n";
    cout << "Fichier maillage : " << meshFile <<  endl;
    MeshReader reader(&mesh);
    reader.setMaterial(1, &material);
    reader.readGmshFile(meshFile);
    mesh.initializeElements();
    mesh.computeGeometry();
    mesh.scaleCoordinates();
}

void printConvergenceTable(const vector<ConvergenceResult>& results, const string& label, const ReportOptions& opt, 
    const function<double(const ConvergenceResult&)>& orderMetric) {
    cout << "\nTableau de convergence (" << label << ") :\n\n";
    cout << defaultfloat << setprecision(8);
    cout << right << setw(14) << "h"
         << setw(10) << "N_elem";
        if (opt.showL2)     cout << setw(16) << "L2_rel";
    if (opt.showDeltaW) cout << setw(16) << "DeltaW_rel";
    cout << " " << setw(10) << "order";
    if (opt.showErrE)   cout << " " << setw(14) << "err_E_rel";
    if (opt.showErrNu)  cout << " " << setw(14) << "err_nu_rel";
    if (opt.showErrG)   cout << " " << setw(14) << "err_G_rel";
    cout << "\n" << string(90, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        double mCur = orderMetric(r);
        double mPrev = (i >= 1) ? orderMetric(results[i-1]) : -1.0;
        const bool hasOrder = (i >= 1 && mPrev > 0.0 && mCur > 0.0 && results[i-1].h > r.h);

        cout << right << setw(14) << r.h
             << setw(10) << r.nElems;
        if (opt.showL2)     cout << setw(16) << r.L2rel;
        if (opt.showDeltaW) cout << setw(16) << r.deltaWrel;
        cout << " ";
        if (hasOrder)
            cout << setw(10) << log(mPrev / mCur) / log(results[i-1].h / r.h);
        else
            cout << setw(10) << "-";
        if (opt.showErrE) {
            cout << " ";
            if (r.errErel >= 0.0) cout << setw(14) << r.errErel;
            else                  cout << setw(14) << "-";
        }
        if (opt.showErrNu) {
            cout << " ";
            if (r.errNurel >= 0.0) cout << setw(14) << r.errNurel;
            else                   cout << setw(14) << "-";
        }
        if (opt.showErrG) {
            cout << " ";
            if (r.errGrel >= 0.0) cout << setw(14) << r.errGrel;
            else                  cout << setw(14) << "-";
        }
        cout << "\n";
    }
    cout << "\n";
}

void exportConvergenceCSV(const vector<ConvergenceResult>& results, const string& path, const ReportOptions& opt, 
    const function<double(const ConvergenceResult&)>& orderMetric) {
    ofstream csv(path);
    if (!csv.is_open()) { cerr << "Impossible d'écrire: " << path << "\n"; return; }
    csv << defaultfloat << setprecision(8);
    csv << "h,n_elems";
    if (opt.showL2)     csv << ",L2_rel";
    if (opt.showDeltaW) csv << ",DeltaW_rel";
    csv << ",order";
    if (opt.showErrE)   csv << ",err_E_rel";
    if (opt.showErrNu)  csv << ",err_nu_rel";
    if (opt.showErrG)   csv << ",err_G_rel";
    csv << "\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        double mCur = orderMetric(r);
        double mPrev = (i >= 1) ? orderMetric(results[i-1]) : -1.0;
        const bool hasOrder = (i >= 1 && mPrev > 0.0 && mCur > 0.0 && results[i-1].h > r.h);

        csv << r.h << "," << r.nElems;
        if (opt.showL2)     csv << "," << r.L2rel;
        if (opt.showDeltaW) csv << "," << r.deltaWrel;
        csv << ",";
        if (hasOrder)
            csv << log(mPrev / mCur) / log(results[i-1].h / r.h);
        else
            csv << "-";
        if (opt.showErrE) { 
            csv << ",";
            if (r.errErel >= 0.0) csv << r.errErel;
            else                  csv << "-";
        }
        if (opt.showErrNu) {
            csv << ",";
            if (r.errNurel >= 0.0) csv << r.errNurel;
            else                   csv << "-";
        }
        if (opt.showErrG) {
            csv << ",";
            if (r.errGrel >= 0.0) csv << r.errGrel;
            else                  csv << "-";
        }
        csv << "\n";
    }
    cout << "Fichier CSV de convergence exporté : " << path << "\n" << endl;
}

string convergenceVtkPath(const Config& config, double lc) {
    namespace fs = std::filesystem;
    const fs::path basePath(config.vtkPath(config.meshFile));
    ostringstream oss;
    oss << basePath.stem().string() << "_lc" << setprecision(8) << lc << ".vtk";
    return (basePath.parent_path() / oss.str()).string();
}

void applyShearDirichletBC(Solver& solver, const Mesh& mesh, double gammaTarget) {
    vector<int> boundaryNodes;
    boundaryNodes.reserve(mesh.leftNodes.size() + mesh.rightNodes.size() + mesh.topNodes.size() + mesh.bottomNodes.size());
    boundaryNodes.insert(boundaryNodes.end(), mesh.leftNodes.begin(), mesh.leftNodes.end());
    boundaryNodes.insert(boundaryNodes.end(), mesh.rightNodes.begin(), mesh.rightNodes.end());
    boundaryNodes.insert(boundaryNodes.end(), mesh.topNodes.begin(), mesh.topNodes.end());
    boundaryNodes.insert(boundaryNodes.end(), mesh.bottomNodes.begin(), mesh.bottomNodes.end());

    sort(boundaryNodes.begin(), boundaryNodes.end());
    boundaryNodes.erase(unique(boundaryNodes.begin(), boundaryNodes.end()), boundaryNodes.end());

    for (int id : boundaryNodes) {
        const double y = mesh.getNode(id).coords.y();
        solver.setDirichletBC(id, 0, gammaTarget * y);
        solver.setDirichletBC(id, 1, 0.0);
    }
}

vector<ConvergenceResult> runConvergence(
    const vector<string>& meshFiles,
    const vector<double>& meshLc,
    const function<bool(size_t, const string&, double, ConvergenceResult&)>& runOne) {

    vector<ConvergenceResult> results;
    const size_t n = min(meshFiles.size(), meshLc.size());

    for (size_t i = 0; i < n; ++i) {
        ConvergenceResult r{};
        if (runOne(i, meshFiles[i], meshLc[i], r)) {
            results.push_back(r);
        }
    }

    sort(results.begin(), results.end(),
         [](const ConvergenceResult& a, const ConvergenceResult& b){ return a.h > b.h; });
    return results;
}
