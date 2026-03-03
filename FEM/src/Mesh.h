#ifndef MESH_H
#define MESH_H

#include <vector>
#include <string>
#include <Eigen/Dense>

class Node {
public:
    int id;
    Eigen::Vector2d coords;

    Node(int nodeId, const Eigen::Vector2d& c) : id(nodeId), coords(c) {}
};

class Element {
public:
    int id;
    std::vector<int> nodeIds;
    class Material* material;
    double area;
    Eigen::Matrix<double, 6, 6> Ke;

    Element(int elemId, const std::vector<int>& nIds, Material* mat);
    
    void computeArea(const Node& n1, const Node& n2, const Node& n3);
    void computeKe(const Node& n1, const Node& n2, const Node& n3);
};

class Mesh {
public:
    std::vector<Node> nodes;
    std::vector<Element> elements;
    
    // Informations géométriques
    double xMin, xMax, yMin, yMax;
    std::vector<int> leftNodes, rightNodes, topNodes, bottomNodes;

    Mesh();

    void addNode(const Node& node) { nodes.push_back(node); }
    void addElement(const Element& elem) { elements.push_back(elem); }

    Node& getNode(int id);
    int nbNodes() const { return nodes.size(); }
    int nbElements() const { return elements.size(); }
    double width() const { return xMax - xMin; }
    double height() const { return yMax - yMin; }
    
    void loadFromGmsh(const std::string& filename);
    void initializeElements();
    void computeGeometry();
    void scaleCoordinates(double scale);  // Applique un facteur d'échelle aux coordonnées
    
    std::vector<int> findNodesAtY(double y, double tol = 1e-6) const;
};

#endif
