#ifndef MESH_READER_H
#define MESH_READER_H

#include "mesh.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include <fstream>

// Structure pour stocker les informations d'une arête
struct Edge {
    int node1, node2;
    int tag;  // 11 pour fibre-matrice, 12 pour bord
    
    Edge(int n1, int n2, int t);
    
    bool operator<(const Edge& other) const;
};

class MeshReader {
private:
    Mesh* mesh;
    std::map<int, Material*> materialMap;  // tag -> Material
    std::set<Edge> edges;
    
    // Méthodes privées pour la lecture
    void readNodes(std::ifstream& file);
    void readElements(std::ifstream& file);
public:
    MeshReader(Mesh* m);
    
    // Associer un matériau à un tag physique
    void setMaterial(int tag, Material* mat);
    
    // Lire le fichier Gmsh
    void readGmshFile(const std::string& filename);
    
    // Accéder aux arêtes
    const std::set<Edge>& getEdges() const;
    std::vector<Edge> getEdgesByTag(int tag) const;
};

#endif