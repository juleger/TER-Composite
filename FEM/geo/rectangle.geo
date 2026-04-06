L = 1.0;
H = 1.0;
lc = lc;

// Calcul du nombre d’éléments à partir de lc
Nx = Ceil(L/lc);
Ny = Ceil(H/lc);

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
Physical Line("left", 10) = {4};
Physical Line("right", 11) = {2};
Physical Line("bottom", 12) = {1};
Physical Line("top", 13) = {3};
Physical Surface("material", 1) = {1};

// Maillage structuré régulier à partir de lc
Transfinite Line {1,3} = Nx + 1 Using Progression 1;
Transfinite Line {2,4} = Ny + 1 Using Progression 1;
Transfinite Surface {1};
Recombine Surface {1};  // quadrangles réguliers

// Ordre des éléments
Mesh.ElementOrder = 1; // Q1 linéaire, mettre 2 pour Q2

// Générer le maillage 2D
Mesh 2;