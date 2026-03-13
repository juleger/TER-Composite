#include "mesh.h"
#include "meshReader.h"
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

Mesh::Mesh() : xMin(0), xMax(0), yMin(0), yMax(0) {}

Node& Mesh::getNode(int id) {
    for (auto& node : nodes)
        if (node.id == id) return node;
    cerr << "Erreur : noeud " << id << " introuvable!" << endl;
    return nodes[0];
}

const Node& Mesh::getNode(int id) const {
    for (const auto& node : nodes)
        if (node.id == id) return node;
    cerr << "Erreur : noeud " << id << " introuvable!" << endl;
    return nodes[0];
}

void Mesh::loadFromGmsh(const string& filename) {
    MeshReader reader(this);
    reader.readGmshFile(filename);
    
    // Initialiser les éléments (calcul de l'aire et Ke)
    initializeElements();

    cout << "Maillage chargé : " << nbNodes() << " noeuds, " << nbElements() << " éléments." << endl;
}

void Mesh::initializeElements() {
    for (auto& elem : elements) {
        elem->linkNodes(nodes);
        elem->compute();
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

    cout << "Dimensions: " << width() << " x " << height() << " m\n" << endl;
    
    cout << "Noeuds: " << nbNodes() << ", Eléments: " << nbElements() << endl;
}

vector<int> Mesh::findNodesAtY(double y, double tol) const {
    if (tol < 0) tol = (yMax - yMin) * 1e-4;
    vector<int> result;
    for (const auto& node : nodes) {
        if (abs(node.coords.y() - y) < tol) {
            result.push_back(node.id);
        }
    }
    return result;
}

vector<int> Mesh::findNodesAtX(double x, double tol) const {
    if (tol < 0) tol = (xMax - xMin) * 1e-4;
    vector<int> result;
    for (const auto& node : nodes) {
        if (abs(node.coords.x() - x) < tol) {
            result.push_back(node.id);
        }
    }
    return result;
}

double Mesh::computeVolumeFraction(Material* mat) const {
    double totalArea = 0.0, matArea = 0.0;
    for (const auto& elem : elements) {
        totalArea += elem->area;
        if (elem->material == mat) matArea += elem->area;
    }
    return matArea / totalArea;
}

void Mesh::scaleCoordinates() {
    // TODO
}
