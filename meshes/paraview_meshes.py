from paraview.simple import *
import glob
import os

files = glob.glob("*.vtk")

for f in files:
    # Charger le mesh
    mesh = OpenDataFile(f)

    view = CreateRenderView()
    
    # --- SURFACE (fond semi-transparent)
    surface_display = Show(mesh, view)
    surface_display.Representation = 'Surface'
    surface_display.Opacity = 0.8

    # --- EDGES (extraction)
    edges = ExtractEdges(Input=mesh)
    edge_display = Show(edges, view)
    edge_display.Representation = 'Wireframe'
    edge_display.LineWidth = 2.0

    # --- CAMÉRA
    view.ResetCamera()
    camera = GetActiveCamera()
    camera.Zoom(1.3)
    # --- FOND BLANC (rapport)
    view.Background = [1, 1, 1]

    # --- RENDU "PLEIN ÉCRAN"
    view.ViewSize = [2000, 1800]

    # --- SAUVEGARDE
    output_name = os.path.splitext(f)[0] + ".png"
    SaveScreenshot(output_name, view, ImageResolution=[2000, 1800])

    # Cleanup (important en batch)
    Delete(edges)
    Delete(mesh)
    Delete(view)