#include "Mesh.h"
#include "MeshReader.h"
#include "Material.h"
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

Element::Element(int elemId, const vector<int>& nIds, Material* mat)
    : id(elemId), nodeIds(nIds), material(mat), area(0.0) {
    Ke.setZero();
}

void Element::computeArea(const Node& n1, const Node& n2, const Node& n3) {
    Vector2d p1 = n1.coords;
    Vector2d p2 = n2.coords;
    Vector2d p3 = n3.coords;
    
    area = 0.5 * abs((p2.x() - p1.x()) * (p3.y() - p1.y()) - 
                     (p3.x() - p1.x()) * (p2.y() - p1.y()));
}

void Element::computeKe(const Node& n1, const Node& n2, const Node& n3) {
    Vector2d p1 = n1.coords;
    Vector2d p2 = n2.coords;
    Vector2d p3 = n3.coords;
    
    if (area < 1e-12) {
        cerr << "Attention : élément " << id << " dégénéré (aire ~ 0)" << endl;
        Ke.setZero();
        return;
    }
    
    // Matrice B pour élément triangulaire P1
    Matrix<double, 3, 6> B;
    B.setZero();
    
    double b1 = p2.y() - p3.y();
    double b2 = p3.y() - p1.y();
    double b3 = p1.y() - p2.y();
    
    double c1 = p3.x() - p2.x();
    double c2 = p1.x() - p3.x();
    double c3 = p2.x() - p1.x();
    
    B(0, 0) = b1;  B(0, 2) = b2;  B(0, 4) = b3;
    B(1, 1) = c1;  B(1, 3) = c2;  B(1, 5) = c3;
    B(2, 0) = c1;  B(2, 2) = c2;  B(2, 4) = c3;
    B(2, 1) = b1;  B(2, 3) = b2;  B(2, 5) = b3;
    
    B /= (2.0 * area);
    
    // Matrice de rigidité du matériau
    Matrix3d C = material->getC();
    
    // Matrice de rigidité élémentaire
    Ke = area * B.transpose() * C * B;
}

Mesh::Mesh() : xMin(0), xMax(0), yMin(0), yMax(0) {}

Node& Mesh::getNode(int id) {
    for (auto& node : nodes) {
        if (node.id == id) return node;
    }
    cerr << "Erreur : noeud " << id << " introuvable!" << endl;
    return nodes[0];
}

void Mesh::loadFromGmsh(const string& filename) {
    MeshReader reader(this);
    reader.readGmshFile(filename);
    
    // Initialiser les éléments (calcul de l'aire et Ke)
    initializeElements();
}

void Mesh::initializeElements() {
    for (auto& elem : elements) {
        Node& n1 = getNode(elem.nodeIds[0]);
        Node& n2 = getNode(elem.nodeIds[1]);
        Node& n3 = getNode(elem.nodeIds[2]);
        
        elem.computeArea(n1, n2, n3);
        if (elem.area > 1e-12) {
            elem.computeKe(n1, n2, n3);
        }
    }
}

void Mesh::computeGeometry() {
    if (nodes.empty()) return;
    
    // Calculer les limites
    Vector2d first = nodes[0].coords;
    xMin = xMax = first.x();
    yMin = yMax = first.y();
    
    for (const auto& node : nodes) {
        xMin = min(xMin, node.coords.x());
        xMax = max(xMax, node.coords.x());
        yMin = min(yMin, node.coords.y());
        yMax = max(yMax, node.coords.y());
    }
    
    // Identifier les nœuds de bord
    leftNodes.clear();
    rightNodes.clear();
    topNodes.clear();
    bottomNodes.clear();
    
    for (const auto& node : nodes) {
        if (abs(node.coords.x() - xMin) < 1e-6) leftNodes.push_back(node.id);
        if (abs(node.coords.x() - xMax) < 1e-6) rightNodes.push_back(node.id);
        if (abs(node.coords.y() - yMin) < 1e-6) bottomNodes.push_back(node.id);
        if (abs(node.coords.y() - yMax) < 1e-6) topNodes.push_back(node.id);
    }
}

vector<int> Mesh::findNodesAtY(double y, double tol) const {
    vector<int> result;
    for (const auto& node : nodes) {
        if (abs(node.coords.y() - y) < tol) {
            result.push_back(node.id);
        }
    }
    return result;
}

void Mesh::scaleCoordinates(double scale) {
    if (abs(scale - 1.0) < 1e-12) return;  // Pas de scaling nécessaire
    
    // Scaler toutes les coordonnées des nœuds
    for (auto& node : nodes) {
        node.coords *= scale;
    }
    
    // Mettre à jour les limites géométriques
    xMin *= scale;
    xMax *= scale;
    yMin *= scale;
    yMax *= scale;
    
    // Recalculer les aires et matrices de rigidité des éléments
    initializeElements();
}