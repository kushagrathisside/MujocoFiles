import os
os.environ["MUJOCO_GL"] = "glfw"
os.environ["XDG_SESSION_TYPE"] = "x11"

import mujoco
from mujoco import viewer
import numpy as np
import time
from robot_descriptions.loaders.mujoco import load_robot_description

class PDController:
    def __init__(self, model, kp=100.0, kd=10.0):
        self.model = model
        self.kp = kp
        self.kd = kd
        
    def compute(self, data, target_pos):
        error = target_pos - data.qpos[:7]
        return self.kp * error - self.kd * data.qvel[:7]

if __name__ == "__main__":
    model = load_robot_description("panda_mj_description")
    data = mujoco.MjData(model)
    controller = PDController(model)
    
    target = np.array([0.0, -1.0, 0.0, -1.5, 0.0, 1.5, 0.5])
    duration = 60  # 60 seconds
    start_time = time.time()
    
    with viewer.launch_passive(model, data) as v:
        while time.time() - start_time < duration:
            data.ctrl[:7] = controller.compute(data, target)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.001)