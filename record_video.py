import os
os.environ["MUJOCO_GL"] = "glfw"
os.environ["XDG_SESSION_TYPE"] = "x11"

import mujoco
from mujoco import viewer
import imageio
import time
from robot_descriptions.loaders.mujoco import load_robot_description

def record_simulation(output_file="simulation.mp4", duration=60, fps=30):
    model = load_robot_description("panda_mj_description")
    data = mujoco.MjData(model)
    
    frames = []
    start_time = time.time()
    
    with viewer.launch_passive(model, data) as v:
        while time.time() - start_time < duration:
            mujoco.mj_step(model, data)
            v.sync()
            frames.append(v.render().copy())
            if len(frames) >= duration * fps:
                break
    
    imageio.mimsave(output_file, frames, fps=fps)

if __name__ == "__main__":
    record_simulation()