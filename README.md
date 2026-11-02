# Humanoid Kick

Code for the paper "Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots".

## Installation

Follow these steps to set up your environment:

1. Create an environment with Python 3.8:

    ```sh
    $ conda create --name <env_name> python=3.8
    $ conda activate <env_name>
    ```

2. Install PyTorch with CUDA support:

    ```sh
    $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

3. Install Isaac Gym

    Download Isaac Gym from [NVIDIA’s website](https://developer.nvidia.com/isaac-gym/download).

    Extract and install:

    ```sh
    $ tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    $ cd isaacgym/python
    $ pip install -e .
    ```

    Configure the environment to handle shared libraries, otherwise cannot found shared library of `libpython3.8`:

    ```sh
    $ cd $CONDA_PREFIX
    $ mkdir -p ./etc/conda/activate.d
    $ vim ./etc/conda/activate.d/env_vars.sh  # Add the following line
    export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
    $ mkdir -p ./etc/conda/deactivate.d
    $ vim ./etc/conda/deactivate.d/env_vars.sh  # Add the following line
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
    unset OLD_LD_LIBRARY_PATH
    ```

4. Install Pinocchio:

    ```sh
    $ conda install pinocchio -c conda-forge
    ```

5. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

## Usage

### 1. Training

To start training a policy, run the following command:

```sh
$ python train.py --task=T1 --headless
```

Training logs and saved models will be stored in `logs/<date-time>/`.

#### Configurations

Training settings are loaded from `envs/<task>.yaml`. You can also override config values using command-line arguments:

- `--checkpoint`: Path of the model checkpoint to load (set to `-1` to use the most recent model).
- `--headless`: Run headless without creating a viewer window.
- `--num_envs`: Number of environments to create.
- `--device`: Device for simulation and training (e.g., `cuda:0`, `cpu`). 
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.

To add a new task, create a config file in `envs/` and register the environment in `envs/__init__.py`.

#### Progress Tracking

To visualize training progress with [TensorBoard](https://www.tensorflow.org/tensorboard), run:

```sh
$ tensorboard --logdir logs
```

To use [Weights & Biases](https://wandb.ai/) for tracking, log in first:

```sh
$ wandb login
```

You can disable W&B tracking by setting `use_wandb` to `false` in the config file.

---

### 2. Playing

#### In-Simulation Testing

To test the trained policy in Isaac Gym, run:

```sh
$ python play.py --task=T1 --checkpoint=-1
```

#### Cross-Simulation Testing

To test the policy in MuJoCo, run:

```sh
$ python play_mujoco.py --task=T1 --checkpoint=-1
```

## Acknowledgments

We use some code from the [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) repositories in the development of this project.
