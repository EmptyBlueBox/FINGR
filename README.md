# FINGR

**Learning Dexterous Hand Control for Real-World Rubik's Cube Solving**

![FINGR](assets/teaser.png)

Install [uv](https://docs.astral.sh/uv/). Training requires Linux and an NVIDIA CUDA GPU.

```bash
git clone https://github.com/EmptyBlueBox/FINGR.git
cd FINGR
uv sync
uv run hf auth login
uv run -m fingr.visualize_training_traj
uv run -m fingr.train_policy
```

[Training data](https://huggingface.co/datasets/EmptyBlue/FINGR) downloads automatically; login is required while the dataset is private. Training uses the paper settings and saves metrics and checkpoints under `checkpoint/`.
