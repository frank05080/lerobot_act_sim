from pathlib import Path

import asyncio
import gymnasium as gym
import gym_aloha
import imageio
import numpy as np
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy

OUTPUT_VIDEO_PATH = "/home/ros/share_dir/gitrepos/lerobot_act/lerobot/outputs/eval/act_aloha_sim_transfer_cube_human"
WEIGHT_PATH = "/home/ros/share_dir/gitrepos/lerobot_act/lerobot/outputs/train/act_aloha_sim_transfer_cube_human"

async def main():
    output_directory = Path(OUTPUT_VIDEO_PATH)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    policy = ACTPolicy.from_pretrained(WEIGHT_PATH, map_location=device)

    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=500
    )

    policy.reset()
    numpy_observation, info = env.reset(seed=42)

    rewards = []
    frames = []

    frames.append(env.render())

    step = 0
    done = False


    while not done:
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"]['top'])

        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        observation = {
            "observation.state": state,
            "observation.images.top": image,
        }

        with torch.inference_mode():
            action = await policy.select_action_bpu(observation)

        numpy_action = action.squeeze(0).to("cpu").numpy()
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated | truncated | done
        step += 1


    if terminated:
        print("Success!")
    else:
        print("Failure!")

    fps = env.metadata["render_fps"]

    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")
    
if __name__ == '__main__':
    asyncio.run(main())