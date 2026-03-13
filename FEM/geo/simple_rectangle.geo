// Rectangle simple pour test de traction
// Dimensions: 10 x 2 (L x H)

L = 10.0;
H = 2.0;
lc = 0.5;  // taille de maille

// Points
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

// Lignes
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Tags physiques
Physical Line("left", 10) = {4};    // Bord gauche (encastré)
Physical Line("right", 11) = {2};   // Bord droit (traction)
Physical Line("bottom", 12) = {1};  // Bord bas
Physical Line("top", 13) = {3};     // Bord haut
Physical Surface("material", 1) = {1};  // Matériau unique

// Générer le maillage
Mesh 2;
