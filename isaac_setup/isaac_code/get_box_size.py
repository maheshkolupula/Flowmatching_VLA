import sys
sys.path.append("/home/emagesoft/Downloads/vlaflow/isaac_setup/isaac_code")

from isaacsim.simulation_app import SimulationApp
app = SimulationApp({"headless": True})

from runtime.isaac_runtime import IsaacRuntime
rt = IsaacRuntime(
    map_joint_names=[],
    sim_app=app,
    usd_path="scene/tomo_packing_suction.usd",
    robot_root="/tomo_re_uhand/Geometry",
    camera_prim="",
    box_prim="/Iphone",
    container_prim="/tomo_re_uhand/Box_1_v1___open"
)

box = rt.bbox_world("/tomo_re_uhand/Box_1_v1___open")
print(f"BBOX: {box}")
if box is not None:
    size = [box[1][0]-box[0][0], box[1][1]-box[0][1], box[1][2]-box[0][2]]
    print(f"BOX DIMENSIONS: X={size[0]:.4f}m, Y={size[1]:.4f}m, Z={size[2]:.4f}m")
else:
    print("Could not get bounding box.")
    
app.close()
