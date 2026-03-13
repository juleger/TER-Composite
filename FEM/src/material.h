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
    
    Eigen::Matrix3d C; // Matrice de rigidité en contraintes planes
    Material(double E_val, double nu_val, double rho_val);
    
    // Calcule la matrice de rigidité en contraintes planes
    void computeC();
};

class CompositeMaterial {
public:
    // Matériau orthotrope, avec propriétés effectives à déterminer
    Material* fiber;
    Material* matrix;

    double V_fiber;  // Fraction volumique de fibre
    double E1, E2, E3;  // Modules de Young effectifs
    double v12, v13, v23;  // Coefficients de Poisson effectifs
    double G12, G13, G23;  // Modules de cisaillement effectifs
    
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero(); // Matrice de compliance
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero(); // Matrice de rigidité

    void computeEffectiveProperties();
    void printProperties() const;
};
#endif
