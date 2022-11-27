import numpy as np
import pandas as pd
import gym

class GymWrapper(gym.Wrapper):
    def __init__(self, env, out_csv_name='results/rewards'):
        self.out_csv_name = out_csv_name
        self.metrics = []
        self.run = 0
        self.step_index = 0

        super(GymWrapper, self).__init__(env)
        

    def reset(self):
        """
        Reset the environment 
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        self.step_index +=1
        obs, reward, done, success = self.env.step(action)
        info = self._compute_step_info(success, reward)
        self.metrics.append(info)
        
        if done:
            self.save_csv(self.out_csv_name, self.run)
            self.run += 1
            self.metrics = []
            self.step_index = 0
        return obs, reward, done, success
    
    def _compute_step_info(self, success, reward):
        return {
            'current_step': self.step_index,
            'success': success,
            'Reward': reward,
        }
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    def close(self):
        pass

objects = [ 
            'button_top', 
            'button_side', 
            'coffee_button', 
            'handle_press_top',
            'handle_press_side',
            'door_lock',
            'door_unlock',
            'dial_turn',
            'faucet_open',
            'faucet_close',
            'window_open',
            'window_close',
            'peg_unplug',
        ]

obj2grp = {
            'button_top': 'button_top', 
            'button_side': 'button_side', 
            'coffee_button': 'coffee_button', 
            'handle_press_top': 'handle_press_top',
            'handle_press_side': 'handle_press_side',
            'door_lock': 'door',
            'door_unlock': 'door',
            'dial_turn': 'dial_turn',
            'faucet_open': 'faucet',
            'faucet_close': 'faucet',
            'window_open': 'window',
            'window_close': 'window',
            'peg_unplug': 'peg_unplug',
        }

def enable_gpu_rendering():
    # This is a hack to enable GPU rendering.
    from dm_control import mujoco
    # Load a model from an MJCF XML string.
    xml_string = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1.5"/>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="box" pos="0 0 .3">
          <joint name="up_down" type="slide" axis="0 0 1"/>
          <geom name="box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
          <geom name="sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    physics = mujoco.Physics.from_xml_string(xml_string)
    # Render the default camera view as a numpy array of pixels.
    pixels = physics.render()

