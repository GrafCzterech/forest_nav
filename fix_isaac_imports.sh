#!/bin/bash
pip install conda-build
conda-develop ~/.conda/envs/env_isaaclab/lib/python3.10/site-packages/isaaclab/source/isaaclab/
conda-develop ~/.conda/envs/env_isaaclab/lib/python3.10/site-packages/isaaclab/source/isaaclab_assets
conda-develop ~/.conda/envs/env_isaaclab/lib/python3.10/site-packages/isaaclab/source/isaaclab_tasks
conda-develop ~/.conda/envs/env_isaaclab/lib/python3.10/site-packages/isaaclab/source/isaaclab_rl
