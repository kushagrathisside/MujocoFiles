import os
os.environ["MUJOCO_GL"] = "glfw"
os.environ["XDG_SESSION_TYPE"] = "x11"

import mujoco
from mujoco import viewer
import time
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("panda_mj_description")
data = mujoco.MjData(model)

duration = 60  # 60 seconds
start_time = time.time()

with viewer.launch_passive(model, data) as v:
    while time.time() - start_time < duration:
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.001)  # Small delay to prevent CPU overload