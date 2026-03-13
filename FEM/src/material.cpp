#include "material.h"
#include <iostream>
using namespace Eigen;
using namespace std;

Material::Material(double E_val, double nu_val, double rho_val)
    : E(E_val), nu(nu_val), rho(rho_val) {
    computeC();
}

void Material::computeC() {
    // Matrice de rigidité pour matériau isotrope en contraintes planes
    double factor = E / (1.0 - nu * nu);
    C(0, 0) = factor;
    C(0, 1) = factor * nu;
    C(0, 2) = 0.0;
    
    C(1, 0) = factor * nu;
    C(1, 1) = factor;
    C(1, 2) = 0.0;
    
    C(2, 0) = 0.0;
    C(2, 1) = 0.0;
    C(2, 2) = factor * (1.0 - nu) / 2.0;
}

void CompositeMaterial::printProperties() const {
    cout << "=== Propriétés du matériau composite ===" << endl;
    cout << "E_1 = " << E1 << " Pa" << endl;
    cout << "E_2 = " << E2 << " Pa" << endl;
    cout << "E_3 = " << E3 << " Pa" << endl;
    cout << "v_12 = " << v12 << endl;
    cout << "v_13 = " << v13 << endl;
    cout << "v_23 = " << v23 << endl;
    cout << "G_12 = " << G12 << " Pa" << endl;
    cout << "G_13 = " << G13 << " Pa" << endl;
    cout << "G_23 = " << G23 << " Pa" << endl;
}

