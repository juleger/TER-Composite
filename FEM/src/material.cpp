#include "material.h"
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace Eigen;
using namespace std;

Material::Material(double E_val, double nu_val, double rho_val)
    : E(E_val), nu(nu_val), rho(rho_val) {
    computeD();
}

void Material::computeD() {
    // Matrice de sigma = D * eps en contrainte plane
    double factor = E / (1.0 - nu * nu);
    D(0, 0) = factor;
    D(0, 1) = factor * nu;
    D(0, 2) = 0.0;
    
    D(1, 0) = factor * nu;
    D(1, 1) = factor;
    D(1, 2) = 0.0;
    
    D(2, 0) = 0.0;
    D(2, 1) = 0.0;
    D(2, 2) = factor * (1.0 - nu) / 2.0;
}

CompositeMaterial::CompositeMaterial(Material* matrixMat, Material* fiberMat, Material* poreMat) {
    setConstituents(matrixMat, fiberMat, poreMat);
}

void CompositeMaterial::setConstituents(Material* matrixMat, Material* fiberMat, Material* poreMat) {
    matrix = matrixMat;
    fiber = fiberMat;
    pore = poreMat;
}

void CompositeMaterial::setVolumeFractions(double fiberFraction, double matrixFraction, double poreFraction) {
    V_fiber = max(0.0, fiberFraction);
    V_matrix = max(0.0, matrixFraction);
    V_pore = max(0.0, poreFraction);
}

void CompositeMaterial::computeVoigtReussBounds() {
    if (matrix == nullptr || fiber == nullptr || pore == nullptr) return;

    E1_voigt = V_fiber * fiber->E + V_matrix * matrix->E + V_pore * pore->E;
    v12_voigt = V_fiber * fiber->nu + V_matrix * matrix->nu + V_pore * pore->nu;

    const double G_fiber = fiber->E / (2.0 * (1.0 + fiber->nu));
    const double G_matrix = matrix->E / (2.0 * (1.0 + matrix->nu));
    const double G_pore = pore->E / (2.0 * (1.0 + pore->nu));
    G12_voigt = V_fiber * G_fiber + V_matrix * G_matrix + V_pore * G_pore;

    const double eps = 1e-30;
    E1_reuss = 1.0 / max(V_fiber / max(fiber->E, eps) + V_matrix / max(matrix->E, eps) + V_pore / max(pore->E, eps), eps);
    v12_reuss = 1.0 / max(V_fiber / max(fiber->nu, eps) + V_matrix / max(matrix->nu, eps) + V_pore / max(pore->nu, eps), eps);
    G12_reuss = 1.0 / max(V_fiber / max(G_fiber, eps) + V_matrix / max(G_matrix, eps) + V_pore / max(G_pore, eps), eps);
}

void CompositeMaterial::updateFromTractionX(double sigmaX, double epsX, double epsY) {
    const double eps = 1e-30;
    E1 = sigmaX / max(abs(epsX), eps);
    v12 = -epsY / max(abs(epsX), eps);
}

void CompositeMaterial::updateFromTractionY(double sigmaY, double epsY, double epsX) {
    const double eps = 1e-30;
    E2 = sigmaY / max(abs(epsY), eps);
    v21 = -epsX / max(abs(epsY), eps);
}

void CompositeMaterial::updateFromShear(double tauXY, double gammaXY) {
    const double eps = 1e-30;
    G12 = tauXY / max(abs(gammaXY), eps);
}

void CompositeMaterial::computeEffectiveProperties() {
    computeVoigtReussBounds();
}

void CompositeMaterial::printProperties() const {
    cout << "=== Propriétés du matériau composite ===" << endl;
    cout << "Vf = " << V_fiber << ", Vm = " << V_matrix << ", Vp = " << V_pore << endl;
    cout << "E_1 = " << E1 << " Pa" << endl;
    cout << "E_2 = " << E2 << " Pa" << endl;
    cout << "E_3 = " << E3 << " Pa" << endl;
    cout << "v_12 = " << v12 << endl;
    cout << "v_21 = " << v21 << endl;
    cout << "v_13 = " << v13 << endl;
    cout << "v_23 = " << v23 << endl;
    cout << "G_12 = " << G12 << " Pa" << endl;
    cout << "G_13 = " << G13 << " Pa" << endl;
    cout << "G_23 = " << G23 << " Pa" << endl;
    cout << "Bornes E1 (Voigt/Reuss) = " << E1_voigt << " / " << E1_reuss << " Pa" << endl;
    cout << "Bornes v12 (Voigt/Reuss) = " << v12_voigt << " / " << v12_reuss << endl;
    cout << "Bornes G12 (Voigt/Reuss) = " << G12_voigt << " / " << G12_reuss << " Pa" << endl;
}

void CompositeMaterial::buildMatrixes() {
    S(0, 0) = 1.0 / max(E1, 1e-30), S(0, 1) = -v12 / max(E1, 1e-30), S(0, 2) = 0.0;
    S(1, 0) = -v21 / max(E2, 1e-30), S(1, 1) = 1.0 / max(E2, 1e-30), S(1, 2) = 0.0;
    S(2, 0) = 0.0, S(2, 1) = 0.0, S(2, 2) = 1.0 / max(G12, 1e-30);
    C = S.inverse();
}
void CompositeMaterial::printC() const {
    cout << "Matrice de rigidité C :" << endl;
    cout << C << endl;
}

void CompositeMaterial::printS() const {
    cout << "Matrice de compliance S :" << endl;
    cout << S << endl;
}