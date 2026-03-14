#ifndef MATERIAL_H
#define MATERIAL_H

#include <Eigen/Dense>
#include <iostream>
// Classe simple pour matériau isotrope
class Material {
public:
    double E;   // Module de Young
    double nu;  // Coefficient de Poisson
    double rho; // Densité
    
    Eigen::Matrix3d D; // Matrice de rigidité en contraintes planes
    Material(double E_val, double nu_val, double rho_val);
    
    // Calcule la matrice de rigidité en contraintes planes
    void computeD();
};

class CompositeMaterial {
public:
    // Constituants
    Material* matrix = nullptr;
    Material* fiber = nullptr;
    Material* pore = nullptr;

    // Fractions volumiques
    double V_fiber = 0.0;
    double V_matrix = 0.0;
    double V_pore = 0.0;

    // Propriétés effectives 
    double E1 = 0.0;
    double E2 = 0.0;
    double E3 = 0.0;
    double v12 = 0.0;
    double v21 = 0.0;
    double v13 = 0.0;
    double v23 = 0.0;
    double G12 = 0.0;
    double G13 = 0.0;
    double G23 = 0.0;

    // Bornes
    double E1_voigt = 0.0;
    double E1_reuss = 0.0;
    double v12_voigt = 0.0;
    double v12_reuss = 0.0;
    double G12_voigt = 0.0;
    double G12_reuss = 0.0;
    
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero(); // Matrice de compliance
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero(); // Matrice de rigidité

    CompositeMaterial() = default;
    CompositeMaterial(Material* matrixMat, Material* fiberMat, Material* poreMat = nullptr);

    void setConstituents(Material* matrixMat, Material* fiberMat, Material* poreMat = nullptr);
    void setVolumeFractions(double fiberFraction, double matrixFraction, double poreFraction = 0.0);
    void computeVoigtReussBounds();

    void updateFromTractionX(double sigmaX, double epsX, double epsY);
    void updateFromTractionY(double sigmaY, double epsY, double epsX);
    void updateFromShear(double tauXY, double gammaXY);

    void buildMatrixes();
    void computeEffectiveProperties();
    void printC() const;
    void printS() const;
    void printProperties() const;
};
#endif
