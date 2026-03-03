#include "MeshReader.h"
#include "Material.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string>

using namespace std;
using namespace Eigen;

Edge::Edge(int n1, int n2, int t) : node1(min(n1, n2)), node2(max(n1, n2)), tag(t) {}

bool Edge::operator<(const Edge& other) const {
    if (node1 != other.node1) return node1 < other.node1;
    return node2 < other.node2;
}

MeshReader::MeshReader(Mesh* m) : mesh(m) {}

void MeshReader::setMaterial(int tag, Material* mat) {
    materialMap[tag] = mat;
}

void MeshReader::readGmshFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erreur : impossible d'ouvrir " << filename << endl;
        return;
    }
    
    string line;
    while (getline(file, line)) {
        if (line.find("$Nodes") != string::npos) {
            readNodes(file);
        }
        else if (line.find("$Elements") != string::npos) {
            readElements(file);
        }
    }
    
    file.close();
    printStatistics();
}

void MeshReader::readNodes(ifstream& file) {
    string line;
    getline(file, line);
    
    int numNodes;
    istringstream iss(line);
    
    // Détecter le format Gmsh
    if (line.find_first_of("0123456789") == 0) {
        vector<string> tokens;
        string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() == 4) {
            // Format Gmsh 4.x : numEntityBlocks numNodes minNodeTag maxNodeTag
            int numEntityBlocks = stoi(tokens[0]);
            numNodes = stoi(tokens[1]);
            
            for (int i = 0; i < numEntityBlocks; i++) {
                getline(file, line);
                istringstream blockHeader(line);
                int entityDim, entityTag, parametric, numNodesInBlock;
                blockHeader >> entityDim >> entityTag >> parametric >> numNodesInBlock;
                
                // Lire les IDs des noeuds
                vector<int> nodeIds;
                for (int j = 0; j < numNodesInBlock; j++) {
                    getline(file, line);
                    nodeIds.push_back(stoi(line));
                }
                
                // Lire les coordonnées
                for (int j = 0; j < numNodesInBlock; j++) {
                    getline(file, line);
                    istringstream coords(line);
                    double x, y, z;
                    coords >> x >> y >> z;
                    
                    Vector2d pos(x, y);
                    mesh->addNode(Node(nodeIds[j], pos));
                }
            }
        }
        else {
            // Format Gmsh 2.2 : simple numNodes
            numNodes = stoi(tokens[0]);
            for (int i = 0; i < numNodes; i++) {
                getline(file, line);
                istringstream nodeStream(line);
                int id;
                double x, y, z;
                nodeStream >> id >> x >> y >> z;
                
                Vector2d pos(x, y);
                mesh->addNode(Node(id, pos));
            }
        }
    }
    
    getline(file, line);  // $EndNodes
}

void MeshReader::readElements(ifstream& file) {
    string line;
    getline(file, line);
    
    int numElements;
    istringstream iss(line);
    
    int elemIdCounter = 1;
    int numTrianglesMatrix = 0;
    int numTrianglesFiber = 0;
    int numEdgesFiberMatrix = 0;
    int numEdgesBoundary = 0;
    
    vector<string> tokens;
    string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    if (tokens.size() == 4) {
        // Format Gmsh 4.x : numEntityBlocks numElements minTag maxTag
        int numEntityBlocks = stoi(tokens[0]);
        numElements = stoi(tokens[1]);
        
        for (int i = 0; i < numEntityBlocks; i++) {
            getline(file, line);
            istringstream blockHeader(line);
            int entityDim, entityTag, elementType, numElementsInBlock;
            blockHeader >> entityDim >> entityTag >> elementType >> numElementsInBlock;
            
            for (int j = 0; j < numElementsInBlock; j++) {
                getline(file, line);
                istringstream elemStream(line);
                int elemId;
                elemStream >> elemId;
                
                if (elementType == 2) {  // Triangle
                    int n1, n2, n3;
                    elemStream >> n1 >> n2 >> n3;
                    
                    vector<int> nodeIds = {n1, n2, n3};
                    Material* mat = materialMap[entityTag];
                    
                    if (mat == nullptr) {
                        cerr << "Attention : matériau non défini pour le tag " 
                                  << entityTag << endl;
                    }
                    
                    mesh->addElement(Element(elemIdCounter++, nodeIds, mat));
                    
                    if (entityTag == 1) numTrianglesMatrix++;
                    else if (entityTag == 2) numTrianglesFiber++;
                }
                else if (elementType == 1) {  // Segment (arête)
                    int n1, n2;
                    elemStream >> n1 >> n2;
                    
                    edges.insert(Edge(n1, n2, entityTag));
                    
                    if (entityTag == 11) numEdgesFiberMatrix++;
                    else if (entityTag == 12) numEdgesBoundary++;
                }
            }
        }
    }
    else {
        // Format Gmsh 2.2
        numElements = stoi(tokens[0]);
        
        for (int i = 0; i < numElements; i++) {
            getline(file, line);
            istringstream elemStream(line);
            
            int elemId, elemType, numTags;
            elemStream >> elemId >> elemType >> numTags;
            
            vector<int> tags;
            for (int t = 0; t < numTags; t++) {
                int tag;
                elemStream >> tag;
                tags.push_back(tag);
            }
            
            int physicalTag = (numTags > 0) ? tags[0] : 0;
            
            if (elemType == 2) {  // Triangle
                int n1, n2, n3;
                elemStream >> n1 >> n2 >> n3;
                
                vector<int> nodeIds = {n1, n2, n3};
                Material* mat = materialMap[physicalTag];
                
                if (mat == nullptr) {
                    cerr << "Attention : matériau non défini pour le tag " 
                              << physicalTag << endl;
                }
                
                mesh->addElement(Element(elemIdCounter++, nodeIds, mat));
                
                if (physicalTag == 1) numTrianglesMatrix++;
                else if (physicalTag == 2) numTrianglesFiber++;
            }
            else if (elemType == 1) {  // Segment (arête)
                int n1, n2;
                elemStream >> n1 >> n2;
                
                edges.insert(Edge(n1, n2, physicalTag));
                
                if (physicalTag == 11) numEdgesFiberMatrix++;
                else if (physicalTag == 12) numEdgesBoundary++;
            }
        }
    }
    
    getline(file, line);  // $EndElements
}

void MeshReader::printStatistics() const {
    // Statistiques simplifiées déjà affichées
}

const set<Edge>& MeshReader::getEdges() const {
    return edges;
}

vector<Edge> MeshReader::getEdgesByTag(int tag) const {
    vector<Edge> result;
    for (const auto& edge : edges) {
        if (edge.tag == tag) {
            result.push_back(edge);
        }
    }
    return result;
}