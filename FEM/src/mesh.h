#ifndef MESH_H
#define MESH_H

#include "element.h"
#include <vector>
#include <string>
#include <memory>

class Mesh {
public:
    std::vector<Node> nodes;
    std::vector<std::unique_ptr<Element>> elements;

    // Informations géométriques
    double xMin, xMax, yMin, yMax;
    std::vector<int> leftNodes, rightNodes, topNodes, bottomNodes;

    Mesh();

    void addNode(const Node& node) { nodes.push_back(node); }
    void addElement(std::unique_ptr<Element> elem) { elements.push_back(std::move(elem)); }

    Node& getNode(int id);
    const Node& getNode(int id) const;
    int nbNodes()    const { return (int)nodes.size(); }
    int nbElements() const { return (int)elements.size(); }
    double width()   const { return xMax - xMin; }
    double height()  const { return yMax - yMin; }

    double computeVolumeFraction(Material* mat) const;
    double computeCharacteristicLength() const;

    void loadFromGmsh(const std::string& filename);
    void initializeElements();
    void computeGeometry();
    void scaleCoordinates();

    std::vector<int> findNodesAtY(double y, double tol = -1) const;
    std::vector<int> findNodesAtX(double x, double tol = -1) const;
};

#endif
