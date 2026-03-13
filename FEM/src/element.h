#ifndef ELEMENT_H
#define ELEMENT_H

#include "material.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>

class Node {
public:
    int id;
    Eigen::Vector2d coords;
    Node(int id, const Eigen::Vector2d& c) : id(id), coords(c) {}
};

class Element {
public:
    int id;
    std::vector<int> nodeIds;
    Material* material;
    double area = 0.0;
    std::vector<const Node*> nodes;
    Eigen::MatrixXd B; // Matrice de déformation (3×2n)
    Eigen::MatrixXd Ke; // Matrice de rigidité élémentaire

    Element(int id, const std::vector<int>& nIds, Material* mat);
    virtual ~Element() = default;

    int nDofs()   const { return 2 * nNodes(); }
    virtual int nNodes()  const = 0;
    virtual int vtkType() const = 0;

    void linkNodes(const std::vector<Node>& allNodes);
    virtual void compute() = 0;  // calcule area, B, Ke
    // Valeurs des fonctions de forme au centroïde de l'élément
    virtual Eigen::VectorXd shapeAtCentroid() const = 0;

protected:
    void computeJacobian(const Eigen::VectorXd& dNdxi, const Eigen::VectorXd& dNdeta, 
        Eigen::Matrix2d& J, Eigen::Matrix2d& invJ, double& detJ) const;
    Eigen::MatrixXd buildB(const Eigen::VectorXd& dNdx, const Eigen::VectorXd& dNdy) const;
};

class ElementP1 : public Element {
public:
    using Element::Element;
    int nNodes()  const override { return 3; }
    int vtkType() const override { return 5; }   // VTK_TRIANGLE
    void compute() override;
    Eigen::VectorXd shapeAtCentroid() const override {
        return Eigen::Vector3d(1.0/3, 1.0/3, 1.0/3);
    }
};

class ElementP2 : public Element {
public:
    using Element::Element;
    int nNodes()  const override { return 6; }
    int vtkType() const override { return 22; }  // VTK_QUADRATIC_TRIANGLE
    void compute() override;
    Eigen::VectorXd shapeAtCentroid() const override {
        Eigen::VectorXd N(6);
        N << -1.0/9, -1.0/9, -1.0/9, 4.0/9, 4.0/9, 4.0/9;
        return N;
    }
};

class ElementQ1 : public Element {
public:
    using Element::Element;
    int nNodes()  const override { return 4; }
    int vtkType() const override { return 9; }   // VTK_QUAD
    void compute() override;
    Eigen::VectorXd shapeAtCentroid() const override {
        return Eigen::Vector4d(0.25, 0.25, 0.25, 0.25);
    }
};

// Codes Gmsh : 2=P1, 9=P2, 3=Q1
std::unique_ptr<Element> makeElement(int gmshType, int id, 
    const std::vector<int>& nIds, Material* mat);

#endif
