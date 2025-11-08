import trimesh
import pyrender

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for offscreen rendering

# Load the OBJ file
mesh = trimesh.load('/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/housecat/obj_models_small_size_final/glass/glass-cocktail.obj')

# Convert to pyrender mesh
render_mesh = pyrender.Mesh.from_trimesh(mesh)

# Create a scene
scene = pyrender.Scene()
scene.add(render_mesh)

# Create offscreen renderer
r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)

# Add a camera
camera = pyrender.PerspectiveCamera(yfov=3.14 / 3.0)
camera_pose = [[1.0, 0.0, 0.0, 0],
               [0.0, 1.0, 0.0, 0],
               [0.0, 0.0, 1.0, 1.0],
               [0.0, 0.0, 0.0, 1.0]]
scene.add(camera, pose=camera_pose)

# Render the scene
color, depth = r.render(scene)

# Save to image file
import imageio
imageio.imwrite('render_obj.png', color)

r.delete()
print("Saved render_obj.png ✅")
