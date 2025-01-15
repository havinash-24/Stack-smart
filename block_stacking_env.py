# Environment Code (block_stacking_env.py)

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class BlockStackingEnv(gym.Env):
    def __init__(self, max_steps=300, max_blocks=10, render_mode=False, difficulty_stage=1):
        super(BlockStackingEnv, self).__init__()
        self.max_steps = max_steps
        self.max_blocks = max_blocks
        self.render_mode = render_mode
        self.difficulty_stage = difficulty_stage
        self.physics_client = None
        self.current_step = 0
        self.stacked_blocks = []

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 360]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.max_blocks * 13) + 2,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.current_step = 0
        self.stacked_blocks = []
        p.loadURDF("plane.urdf")
        return self._get_observation(), {}

    def step(self, action):
        x, y, rotation = action + np.random.uniform(-0.01, 0.01, size=3)
        z = len(self.stacked_blocks) * 0.5 + 0.5
        weight, friction, size = self._generate_block_properties()
        block = p.createMultiBody(
            baseMass=weight,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[size / 2, size / 2, size / 2]),
            basePosition=[x, y, z]
        )
        p.changeDynamics(block, -1, lateralFriction=friction)
        self.stacked_blocks.append(block)
        
        reward = self._calculate_reward()
        done = self._check_termination() or self.current_step >= self.max_steps
        truncated = len(self.stacked_blocks) >= self.max_blocks
        self.current_step += 1
        obs = self._get_observation()
        return obs, reward, done, truncated, {}

    def _get_observation(self):
        observation = []
        for block in self.stacked_blocks:
            position, orientation = p.getBasePositionAndOrientation(block)
            linear_velocity, angular_velocity = p.getBaseVelocity(block)
            observation.extend(list(position) + list(orientation) + list(linear_velocity) + list(angular_velocity))
        while len(observation) < self.max_blocks * 13:
            observation.extend([0] * 13)
        observation.append(self._calculate_stability())
        observation.append(self._calculate_alignment())
        return np.array(observation, dtype=np.float32)

    def _calculate_reward(self):
        if self._check_collapse():
            return -200 if self.current_step < 5 else -100
        height_reward = len(self.stacked_blocks) ** 2
        stability_bonus = self._calculate_stability() * 30
        alignment_bonus = self._calculate_alignment() * len(self.stacked_blocks) * 5
        return height_reward + stability_bonus + alignment_bonus

    def _calculate_stability(self):
        if not self.stacked_blocks:
            return 1.0
        base_pos, _ = p.getBasePositionAndOrientation(self.stacked_blocks[0])
        top_pos, _ = p.getBasePositionAndOrientation(self.stacked_blocks[-1])
        tilt = np.linalg.norm(np.array(top_pos[:2]) - np.array(base_pos[:2]))
        return max(0, 1 - tilt / 0.3)

    def _calculate_alignment(self):
        if len(self.stacked_blocks) < 2:
            return 1.0
        alignment_score = 0
        for i in range(1, len(self.stacked_blocks)):
            prev_pos = p.getBasePositionAndOrientation(self.stacked_blocks[i - 1])[0]
            curr_pos = p.getBasePositionAndOrientation(self.stacked_blocks[i])[0]
            alignment_score += max(0, 1 - abs(curr_pos[0] - prev_pos[0]) / 0.3)
        return alignment_score / (len(self.stacked_blocks) - 1)

    def _generate_block_properties(self):
        if self.difficulty_stage == 1:
            return 1.0, 0.5, 0.2
        elif self.difficulty_stage == 2:
            return np.random.uniform(0.8, 1.2), np.random.uniform(0.4, 0.6), np.random.uniform(0.15, 0.25)
        elif self.difficulty_stage == 3:
            return np.random.uniform(0.6, 1.5), np.random.uniform(0.3, 0.7), np.random.uniform(0.1, 0.3)
        else:
            fragile_chance = 0.3
            weight = np.random.uniform(0.5, 2.0)
            friction = np.random.uniform(0.2, 0.8)
            size = np.random.uniform(0.1, 0.3)
            if np.random.random() < fragile_chance:
                weight *= 0.5
            return weight, friction, size

    def _check_collapse(self):
        for i in range(1, len(self.stacked_blocks)):
            prev_pos = p.getBasePositionAndOrientation(self.stacked_blocks[i - 1])[0]
            curr_pos = p.getBasePositionAndOrientation(self.stacked_blocks[i])[0]
            if abs(curr_pos[0] - prev_pos[0]) > 0.3 or abs(curr_pos[1] - prev_pos[1]) > 0.3:
                return True
        return False

    def _check_termination(self):
        if self.current_step < 5 and self._check_collapse():
            return True
        return self._check_collapse() or len(self.stacked_blocks) >= self.max_blocks

    def render(self):
        pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

# Training Agent Code (train_agent.py)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from block_stacking_env import BlockStackingEnv

stages = [
    {"difficulty_stage": 1, "max_blocks": 20, "timesteps": 7500},
    {"difficulty_stage": 2, "max_blocks": 30, "timesteps": 10000},
    {"difficulty_stage": 3, "max_blocks": 50, "timesteps": 15000},
    {"difficulty_stage": 4, "max_blocks": 55, "timesteps": 20000}
]

ppo_params = {
    "learning_rate": 3e-4,
    "n_steps": 8192,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "policy_kwargs": dict(net_arch=[256, 256])
}

for stage in stages:
    difficulty_stage = stage["difficulty_stage"]
    max_blocks = stage["max_blocks"]
    total_timesteps = stage["timesteps"]

    print(f"Training on Difficulty Stage {difficulty_stage}...")

    env = DummyVecEnv([lambda: Monitor(BlockStackingEnv(
        render_mode=False,
        difficulty_stage=difficulty_stage,
        max_blocks=max_blocks,
        max_steps=300
    ))])
    env = VecNormalize(env, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1, **ppo_params)

    model.learn(total_timesteps=total_timesteps)

    model.save(f"block_stacking_model_stage_{difficulty_stage}")

    eval_env = DummyVecEnv([lambda: Monitor(BlockStackingEnv(
        max_blocks=max_blocks,
        render_mode=False,
        difficulty_stage=difficulty_stage
    ))])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=True)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    print(f"Stage {difficulty_stage} Evaluation: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Completed Stage {difficulty_stage} with max blocks {max_blocks}. Mean reward: {mean_reward:.2f}.")

    env.close()
    eval_env.close()