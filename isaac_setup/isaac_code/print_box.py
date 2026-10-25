import sys
sys.path.append("/home/emagesoft/Downloads/vlaflow/isaac_setup/isaac_code")
from runtime.isaac_runtime import IsaacRuntime
rt = IsaacRuntime(usd_path="scene/tomo_packing_suction.usd", headless=True, gui=False, box_prim="/Iphone")
box = rt.bbox_world("/tomo_re_uhand/Box_1_v1___open")
print(f"BBOX MIN/MAX: {box}")
size = [box[1][0]-box[0][0], box[1][1]-box[0][1], box[1][2]-box[0][2]]
print(f"BOX DIMENSIONS (X, Y, Z meters): {size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}")
rt.close()
