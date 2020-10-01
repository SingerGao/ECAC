# Error Controlled Actor-critic

This is a PyTorch implementation of our new proposed reinforcement learning algorithm called Error Controlled Actor-critic. (The full paper is to be updated)

## Requirements

- Python 3
- [PyTorch](https://pytorch.org/)
- [OpenAI Gym](https://github.com/openai/gym#id3)
- [MuJoCo](http://www.mujoco.org/)

- [PyBullet](https://github.com/openai/gym/blob/master/docs/environments.md#pybullet-robotics-environments)

- [mpi4py](https://github.com/mpi4py/mpi4py)

## Usage



```
mpiexec -n 5 python main.py --cuda --env-name HopperBulletEnv-v0 --num_steps 1000000 --lr 1e-3  --reward_scale 5 --limit_kl --kl_target 5e-3
```

# ECAC
# ECAC
