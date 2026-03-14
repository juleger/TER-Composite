#include "element.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

// Classe de base des différents types d'éléments (P1, P2, Q1, etc.)

Element::Element(int elemId, const vector<int>& nIds, Material* mat)
    : id(elemId), nodeIds(nIds), material(mat) {}

void Element::linkNodes(const vector<Node>& allNodes) {
    nodes.clear();
    nodes.reserve(nodeIds.size());
    for (int nid : nodeIds)
        nodes.push_back(&allNodes[nid - 1]);
}

void Element::computeJacobian(const VectorXd& dNdxi, const VectorXd& dNdeta,
                               Matrix2d& J, Matrix2d& invJ, double& detJ) const {
    J.setZero();
    for (int i = 0; i < (int)nodes.size(); i++) {
        double x = nodes[i]->coords.x(), y = nodes[i]->coords.y();
        J(0,0) += dNdxi(i)  * x;  J(0,1) += dNdxi(i)  * y;
        J(1,0) += dNdeta(i) * x;  J(1,1) += dNdeta(i) * y;
    }
    detJ = J.determinant();
    invJ = J.inverse();
}

MatrixXd Element::buildB(const VectorXd& dNdx, const VectorXd& dNdy) const {
    int n = (int)nodes.size();
    MatrixXd Bg = MatrixXd::Zero(3, 2*n);
    for (int i = 0; i < n; i++) {
        Bg(0, 2*i)   = dNdx(i);
        Bg(1, 2*i+1) = dNdy(i);
        Bg(2, 2*i)   = dNdy(i);
        Bg(2, 2*i+1) = dNdx(i);
    }
    return Bg;
}

// P1 : triangle linéaire

void ElementP1::compute() {
    // Triangle de référence: N1=1-xi-eta, N2=xi, N3=eta
    VectorXd dNdxi(3), dNdeta(3);
    dNdxi  << -1.0, 1.0, 0.0;
    dNdeta << -1.0, 0.0, 1.0;

    Matrix2d J, invJ;
    double detJ;
    computeJacobian(dNdxi, dNdeta, J, invJ, detJ);

    VectorXd dNdx = invJ(0,0) * dNdxi + invJ(0,1) * dNdeta;
    VectorXd dNdy = invJ(1,0) * dNdxi + invJ(1,1) * dNdeta;

    B = buildB(dNdx, dNdy);
    area = 0.5 * abs(detJ);
    Ke = B.transpose() * material->D * B * area;
}

// P2 : triangle quadratique 6 noeuds
void ElementP2::compute() {
    const double pts[3][2] = {{1.0/6, 1.0/6}, {2.0/3, 1.0/6}, {1.0/6, 2.0/3}}; // points de Gauss pour intégrer
    const double gw = 1.0/6.0; // poids quadrature

    Ke   = MatrixXd::Zero(12, 12);
    area = 0.0;

    Matrix2d J, invJ;
    double detJ;

    // Dérivées des fonctions de forme P2
    auto shapeDerivs = [](double xi, double eta, VectorXd& dxi, VectorXd& deta) {
        dxi  << 4*xi+4*eta-3, 4*xi-1, 0, 4-8*xi-4*eta, 4*eta, -4*eta;
        deta << 4*xi+4*eta-3,       0, 4*eta-1,        -4*xi,  4*xi, 4-4*xi-8*eta;
    };

    VectorXd dNdxi(6), dNdeta(6);

    for (int g = 0; g < 3; g++) {
        double xi = pts[g][0], eta = pts[g][1];
        shapeDerivs(xi, eta, dNdxi, dNdeta);
        computeJacobian(dNdxi, dNdeta, J, invJ, detJ);
        VectorXd dNdx = invJ(0,0)*dNdxi + invJ(0,1)*dNdeta;
        VectorXd dNdy = invJ(1,0)*dNdxi + invJ(1,1)*dNdeta;
        MatrixXd Bg   = buildB(dNdx, dNdy);
        Ke   += gw * Bg.transpose() * material->D * Bg * abs(detJ);
        area += gw * abs(detJ);
    }

    shapeDerivs(1.0/3, 1.0/3, dNdxi, dNdeta);
    computeJacobian(dNdxi, dNdeta, J, invJ, detJ);
    B = buildB(invJ(0,0)*dNdxi + invJ(0,1)*dNdeta, invJ(1,0)*dNdxi + invJ(1,1)*dNdeta);
}

// Q1 : quadrilatère bilinéaire 4 noeuds
void ElementQ1::compute() {
    const double gc = 1.0 / sqrt(3.0);
    const double pts[4][2] = {{-gc,-gc},{gc,-gc},{gc,gc},{-gc,gc}};
    const double xi_n[]  = {-1.0, 1.0, 1.0,-1.0};
    const double eta_n[] = {-1.0,-1.0, 1.0, 1.0};

    Ke = MatrixXd::Zero(8, 8);
    area = 0.0;

    Matrix2d J, invJ;
    double detJ;

    VectorXd dNdxi(4), dNdeta(4);

    for (int g = 0; g < 4; g++) {
        double xi = pts[g][0], eta = pts[g][1];
        for (int i = 0; i < 4; i++) {
            dNdxi(i)  = xi_n[i]  * (1.0 + eta_n[i]*eta) / 4.0;
            dNdeta(i) = eta_n[i] * (1.0 + xi_n[i] *xi)  / 4.0;
        }
        computeJacobian(dNdxi, dNdeta, J, invJ, detJ);
        VectorXd dNdx = invJ(0,0)*dNdxi + invJ(0,1)*dNdeta;
        VectorXd dNdy = invJ(1,0)*dNdxi + invJ(1,1)*dNdeta;
        MatrixXd Bg = buildB(dNdx, dNdy);
        Ke += Bg.transpose() * material->D * Bg * abs(detJ);  // poids = 1
        area += abs(detJ);
    }

    // B au centroïde (0, 0)
    for (int i = 0; i < 4; i++) {
        dNdxi(i) = xi_n[i]  / 4.0;
        dNdeta(i) = eta_n[i] / 4.0;
    }
    computeJacobian(dNdxi, dNdeta, J, invJ, detJ);
    B = buildB(invJ(0,0)*dNdxi + invJ(0,1)*dNdeta,
               invJ(1,0)*dNdxi + invJ(1,1)*dNdeta);
}

unique_ptr<Element> makeElement(int gmshType, int id, const vector<int>& nIds, Material* mat) {
    switch (gmshType) {
        case 2: return make_unique<ElementP1>(id, nIds, mat);
        case 9: return make_unique<ElementP2>(id, nIds, mat);
        case 3: return make_unique<ElementQ1>(id, nIds, mat);
        default: return nullptr;
    }
}
