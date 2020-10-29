from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE


class SawyerMultiobject(SawyerXYZEnv):
    def __init__(
            self,
            random_init=False,
            obs_type='plain',
            goal_low=None,
            goal_high=None,
            rotMode='fixed',
            obj_of_int=None,
            **kwargs
    ):
        self.quick_init(locals())
        hand_low=(-0.5, 0.40, 0.05)
        hand_high=(0.5, 1, 0.5)
        obj_low=(-0.1, 0.9, 0.04)
        obj_high=(0.1, 0.9, 0.04)
        # obj_low=(-0.1, 0.7, 0.05)
        # obj_high=(0.1, 0.8, 0.05)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0., 0.55, 0.04])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.obj_of_int = obj_of_int

        if goal_low is None:
            goal_low = self.hand_low
        
        if goal_high is None:
            goal_high = self.hand_high

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        self.random_init = random_init
        self.max_path_length = 150
        self.rotMode = rotMode
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low,)),
                np.hstack((self.hand_high, obj_high,)),
            )
        elif self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            raise NotImplementedError
        self.reset()

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_multiobject.xml')

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pullDist = self.compute_reward(action, obs_dict)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        info = {'reachDist': reachDist, 'goalDist': pullDist, 'epRew' : reward, 'pickRew':None, 'success': float(pullDist <= 0.06)}
        info['goal'] = self._state_goal
        return ob, reward, done, info

    def _get_obs(self):
        if self.obj_of_int == 'drawer':
            hand = self.get_endeff_pos()
            objPos =  self.data.get_geom_xpos('handle')
            flat_obs = np.concatenate((hand, objPos))
            if self.obs_type == 'with_goal_and_id':
                return np.concatenate([
                        flat_obs,
                        self._state_goal,
                        self._state_goal_idx
                    ])
            elif self.obs_type == 'with_goal':
                return np.concatenate([
                        flat_obs,
                        self._state_goal
                    ])
            elif self.obs_type == 'plain':
                return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
            else:
                return np.concatenate([flat_obs, self._state_goal_idx])
        elif self.obj_of_int == 'dial':
            hand = self.get_endeff_pos()
            objPos = self.get_site_pos('dialStart')
            # angle = self.get_angle()
            flat_obs = np.concatenate((hand, objPos))
            if self.obs_type == 'with_goal_and_id':
                return np.concatenate([
                        flat_obs,
                        self._state_goal,
                        self._state_goal_idx
                    ])
            elif self.obs_type == 'with_goal':
                return np.concatenate([
                        flat_obs,
                        self._state_goal
                    ])
            elif self.obs_type == 'plain':
                return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
            else:
                return np.concatenate([flat_obs, self._state_goal_idx])

    def _get_obs_dict(self):
        if self.obj_of_int == 'drawer':
            hand = self.get_endeff_pos()
            objPos =  self.data.get_geom_xpos('handle')
            flat_obs = np.concatenate((hand, objPos))
            return dict(
                state_observation=flat_obs,
                state_desired_goal=self._state_goal,
                state_achieved_goal=objPos,
            )
        elif self.obj_of_int == 'dial':
            hand = self.get_endeff_pos()
            objPos =  self.get_site_pos('dialStart')
            flat_obs = np.concatenate((hand, objPos))
            return dict(
                state_observation=flat_obs,
                state_desired_goal=self._state_goal,
                state_achieved_goal=objPos,
            )

    def _get_info(self):
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('handle')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    

    def _set_obj_xyz_quat(self, pos, angle):
        quat = Quaternion(axis = [0,0,1], angle = angle).elements
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)


    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        # qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        if self.obj_of_int == 'drawer':
            self.objHeight = self.data.get_geom_xpos('handle')[2]
            if self.random_init:
                obj_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
                self.obj_init_pos = obj_pos
                goal_pos = obj_pos.copy()
                goal_pos[1] -= 0.2
                self._state_goal = goal_pos
            self._set_goal_marker(self._state_goal)
            drawer_cover_pos = self.obj_init_pos.copy()
            drawer_cover_pos[2] -= 0.02
            self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.obj_init_pos
            self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = drawer_cover_pos
            self.sim.model.site_pos[self.model.site_name2id('goal')] = self._state_goal
            self.sim.model.body_pos[self.model.body_name2id('dial')] = np.array([-0.3, 0.7, 0.05])
            self._set_obj_xyz(-0.2)
            self.curr_path_length = 0
            self.maxDist = np.abs(self.data.get_geom_xpos('handle')[1] - self._state_goal[1])
            self.target_reward = 1000*self.maxDist + 1000*2
            print(self.sim.model.body_pos[self.model.body_name2id('drawer')])
            print(self.sim.model.body_pos[self.model.body_name2id('drawer_cover')])
            print(self.sim.model.site_pos[self.model.site_name2id('goal')])
            print(self.sim.model.body_pos[self.model.body_name2id('dial')])

        elif self.obj_of_int == 'dial':
            self.obj_init_pos = self.init_config['obj_init_pos']
            if self.random_init:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
                self.obj_init_pos = goal_pos[:3]
                final_pos = goal_pos.copy() + np.array([0, 0.03, 0.03])
                self._state_goal = final_pos

            self.sim.model.body_pos[self.model.body_name2id('drawer')] = np.array([-0.3, 0.5, 0.04], dtype=np.float32)
            drawer_cover_pos = np.array([-0.3, 0.5, 0.04], dtype=np.float32)
            drawer_cover_pos[2] -= 0.02
            self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = drawer_cover_pos
            self.sim.model.body_pos[self.model.body_name2id('dial')] = self.obj_init_pos
            self._set_goal_marker(self._state_goal)
            self.maxPullDist = np.abs(self._state_goal[1] - self.obj_init_pos[1])
            self.curr_path_length = 0

            print(self.sim.model.body_pos[self.model.body_name2id('drawer')])
            print(self.sim.model.body_pos[self.model.body_name2id('drawer_cover')])
            print(self.sim.model.body_pos[self.model.body_name2id('dial')])
            print(self._state_goal)

        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        if self.random_init:
            hand_pos = np.random.uniform(
                self.hand_low,
                self.hand_high,
                size=(self.hand_low.size),
            )
            self.hand_init_pos = hand_pos

        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs):
        if self.obj_of_int == 'drawer':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal[1]

            reachDist = np.linalg.norm(objPos - fingerCOM)

            pullDist = np.abs(objPos[1] - pullGoal)

            # reward = -reachDist - pullDist
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if reachDist < 0.05:
                # pullRew = -pullDist
                pullRew = 1000*(self.maxDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0
            reward = -reachDist + pullRew
          
            return [reward, reachDist, pullDist] 

        elif self.obj_of_int == 'dial':
            if isinstance(obs, dict): 
                obs = obs['state_observation']

            objPos = obs[3:6]

            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            fingerCOM  =  (rightFinger + leftFinger)/2

            pullGoal = self._state_goal

            pullDist = np.abs(objPos[1] - pullGoal[1])# + np.abs(objPos[0] - pullGoal[0])
            reachDist = np.linalg.norm(objPos - fingerCOM)
            # reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            # zDist = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            # if reachDistxy < 0.05: #0.02
            #     reachRew = -reachDist
            # else:
            #     reachRew =  -reachDistxy - zDist
            reachRew = -reachDist

            def reachCompleted():
                if reachDist < 0.05:
                    return True
                else:
                    return False

            if reachCompleted():
                self.reachCompleted = True
            else:
                self.reachCompleted = False

            def pullReward():
                # c1 = 5000 ; c2 = 0.001 ; c3 = 0.0001
                c1 = 1000 ; c2 = 0.001 ; c3 = 0.0001
                # c1 = 10 ; c2 = 0.01 ; c3 = 0.001
                if self.reachCompleted:
                    pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                    pullRew = max(pullRew,0)
                    return pullRew
                else:
                    return 0
                # pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                # pullRew = max(pullRew,0)
                # return pullRew
            # pullRew = -pullDist
            pullRew = pullReward()
            reward = reachRew + pullRew# - actions[-1]/50
            # reward = pullRew# - actions[-1]/50
          
            return [reward, reachDist, pullDist] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = -1         # id of the body to track ()
        # self.viewer.cam.distance = self.model.stat.extent * 1.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
        # self.viewer.cam.lookat[0] += 0.5         # x,y,z offset from the object (works if trackbodyid=-1)
        # self.viewer.cam.lookat[1] += 0.5
        # self.viewer.cam.lookat[2] += 0.5
        # self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        # self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.2
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 0.4
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1