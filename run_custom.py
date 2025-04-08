import os
os.environ["MUJOCO_GL"] = "glfw"
os.environ["XDG_SESSION_TYPE"] = "x11"

import mujoco
from mujoco import viewer
import time

def load_custom_scene():
    from robot_descriptions.loaders.mujoco import load_robot_description
    robot_model = load_robot_description("panda_mj_description")
    scene_model = mujoco.MjModel.from_xml_path("custom_scene.xml")
    
    merged_model = mujoco.MjModel()
    merged_model.merge(robot_model)
    merged_model.merge(scene_model)
    
    return merged_model, mujoco.MjData(merged_model)

if __name__ == "__main__":
    model, data = load_custom_scene()
    duration = 60  # 60 seconds
    start_time = time.time()
    
    with viewer.launch_passive(model, data) as v:
        while time.time() - start_time < duration:
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.001)