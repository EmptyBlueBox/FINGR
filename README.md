<div align="center">

# FINGR: Learning Dexterous Hand Control<br>for Real-World Rubik's Cube Solving

[Yutong Liang](https://www.lyt0112.com/)<sup>1,\*</sup> · [Quanquan Peng](https://pengqq.com/)<sup>1,\*</sup> · [Matthew Kim](https://mattkiim.github.io/)<sup>1,\*</sup> · [Xiaolong Wang](https://xiaolonw.github.io/)<sup>1</sup>

<sup>1</sup>UC San Diego · <sup>\*</sup>Equal contribution

[![Project Page](https://img.shields.io/badge/Project%20Page-FINGR-4b8bbe)](https://www.lyt0112.com/projects/FINGR) [![Video](https://img.shields.io/badge/Video-YouTube-ff0000)](https://youtu.be/0rlplkw3sxQ) [![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e)](https://huggingface.co/datasets/EmptyBlue/FINGR) [![Visualization](https://img.shields.io/badge/Visualization-Interactive-00a8a8)](https://www.lyt0112.com/projects/FINGR#visualization)

</div>

![FINGR](assets/teaser.png)

Install [uv](https://docs.astral.sh/uv/). Training requires Linux and an NVIDIA CUDA GPU.

```bash
git clone https://github.com/EmptyBlueBox/FINGR.git
cd FINGR
uv run -m fingr.visualize_training_traj
uv run -m fingr.train_policy
```

Training data downloads automatically. Training uses the paper settings and saves metrics and checkpoints under `checkpoint/`.
