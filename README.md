<img src="figs/jakiro.webp" alt="Jakiro" width="200" align="left"><div align="center"><h1>&nbsp;Jakiro</h1></div>

<p align="center">
<a href="https://arxiv.org/pdf/"><b>Paper (Jakiro)</b></a>
</p>

<div style="height:20px;"></div>

<p align="center">
  <a href="https://https://github.com/haiduo/Jakiro">
    <img src="https://img.shields.io/badge/Version-v1.0.0-orange.svg" alt="Version">
  </a>
  <a href="https://https://github.com/haiduo/Jakiro/issues">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
  </a>
  <a href="https://https://github.com/haiduo/Jakiro">
    <img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
  </a>
</p>

<div style="height:25px;"></div>

## Speedup ratios of different models on the MT-bench under non-greedy settings.

<p align="center">
  <img src="./figs/A100-40G_MTbench_S_Comparison_Temperature_1.png" alt="benchmark" width="900">
</p>

**Jakiro** is an advanced approach designed to enhance speculative decoding (SD) for large language models. By integrating Mixture of Experts (MoE), Jakiro enables independent experts to generate diverse predictions, effectively decoupling correlations among candidates and addressing a key limitation of traditional tree-based sampling. Jakiro significantly boosts prediction accuracy and inference speed, setting a new state-of-the-art (SOTA) in speculative decoding. Extensive experiments across various models demonstrate its robustness and effectiveness in real-world applications.

## Test demo
The following shows the actual measured inference speeds of Jakiro and EAGLE-2 on a single RTX 4090 GPU with 24GB of memory. As shown, Jakiro has a faster decoding speed and a higher compression ratio.

<div align="center" style="display: flex; justify-content: center;">
  <img src="./figs/eagle2.gif" alt="EAGLE-2 Demo" style="max-width: 49.5%; height: auto; margin-right: 1px;">
  <img src="./figs/jakiro.gif" alt="Jakiro Demo" style="max-width: 49.5%; height: auto; margin-left: 1px;">
</div>

## Code
The code is currently being organized and will be released soon. Stay tuned!

## Reference
For technical details and full experimental results, please check [the paper of Jakiro](https://arxiv.org/abs/2502.06282).
```
@misc{huang2025jakiroboostingspeculativedecoding,
      title={Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE}, 
      author={Haiduo Huang and Fuwei Yang and Zhenhua Liu and Yixing Xu and Jinze Li and Yang Liu and Xuanwu Yin and Dong Li and Pengju Ren and Emad Barsoum},
      year={2025},
      eprint={2502.06282},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.06282}, 
}
```

## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [EAGLE](https://github.com/SafeAILab/EAGLE), [Medusa](https://github.com/FasterDecoding/Medusa), [FastChat](https://github.com/lm-sys/FastChat), and others. The logo is designed by GPT-4o.