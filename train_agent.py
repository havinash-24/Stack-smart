from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from block_stacking_env import BlockStackingEnv

# Stage configurations with curriculum learning
stages = [
    {"difficulty_stage": 1, "max_blocks": 20, "timesteps": 7500},
    {"difficulty_stage": 2, "max_blocks": 30, "timesteps": 10000},
    {"difficulty_stage": 3, "max_blocks": 50, "timesteps": 15000},
    {"difficulty_stage": 4, "max_blocks": 55, "timesteps": 20000}
]

# PPO Parameters (adjusted for better learning stability)
ppo_params = {
    "learning_rate": 3e-4,  # Reduced learning rate for stability
    "n_steps": 8192,  # Larger batch size for more information per update
    "batch_size": 64,  # Batch size for gradient updates
    "n_epochs": 10,  # Number of epochs for each update
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # Generalized Advantage Estimation
    "clip_range": 0.2,  # Conservative updates
    "ent_coef": 0.005,  # Regularization term for exploration
    "policy_kwargs": dict(net_arch=[256, 256])  # Larger network for more complex tasks
}

for stage in stages:
    difficulty_stage = stage["difficulty_stage"]
    max_blocks = stage["max_blocks"]
    total_timesteps = stage["timesteps"]

    print(f"Training on Difficulty Stage {difficulty_stage}...")

    # Create and wrap environment with Monitor for better logging
    env = DummyVecEnv([lambda: Monitor(BlockStackingEnv(
        render_mode=False,
        difficulty_stage=difficulty_stage,
        max_blocks=max_blocks,
        max_steps=300
    ))])
    env = VecNormalize(env, norm_reward=True)

    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **ppo_params
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save the model for the current stage
    model.save(f"block_stacking_model_stage_{difficulty_stage}")

    # Evaluate the agent
    eval_env = DummyVecEnv([lambda: Monitor(BlockStackingEnv(
        max_blocks=max_blocks,
        render_mode=False,
        difficulty_stage=difficulty_stage
    ))])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=True)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    print(f"Stage {difficulty_stage} Evaluation: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Log results and progress
    print(f"Completed Stage {difficulty_stage} with max blocks {max_blocks}. Mean reward: {mean_reward:.2f}.")
    
    # Clean up
    env.close()
    eval_env.close()


