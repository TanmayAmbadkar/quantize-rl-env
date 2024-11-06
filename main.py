# Import the custom environment wrapper and policy
import gymnasium as gym
from ppo.ppo import PPO
from ppo.custom_arch import CustomCNNExtractor, CustomActorCriticPolicy
from stable_baselines3.common.logger import configure
import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import imageio
import numpy as np

def train(env):
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        share_features_extractor=True
    )
    

    model = PPO("CnnPolicy", env, verbose=1, device="cuda", policy_kwargs = policy_kwargs)
    
    new_logger = configure("logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train the model and collect data
    model.learn(total_timesteps=500000)
    model.save("ppo_carracing")


#TESTING CODE
def test(env):
    model = PPO.load("ppo_carracing", env=env, device = "cpu")

    state, _ = env.reset()
    images = []
    recons = []
    while True:
        emb, ind = model.policy.features_extractor.cnn.get_quantized_embedding_with_id(torch.Tensor(state).permute(2, 0, 1).reshape(1, 3, 96, 96)/255)
        state, reward, done, trunc, _ = env.step(env.action_space.sample())
        
        state_tensor = torch.Tensor(state).permute(2, 0, 1).reshape(1, 3, 96, 96)/255
        recon, loss = model.policy.features_extractor.cnn(state_tensor)
        
        recon = to_pil_image(recon[0][0])
        
        images.append(to_pil_image(state_tensor[0]))
        recons.append(recon)
        
        if done or trunc:
            break

    imageio.mimsave(f"gifs/carracing_true.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
    imageio.mimsave(f"gifs/carracing_recon.gif", [np.array(img) for i, img in enumerate(recons) if i%2 == 0], fps=29)
   
if __name__ == "__main__":
    
    env = gym.make("CarRacing-v2")
    test(env)