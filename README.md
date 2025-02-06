<img src="figs/jakiro.png" alt="Jakiro" width="250" align="left"><div align="center"><h1>&nbsp;Jakiro</h1></div>

This repository is the official implementation of "Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE"

<p align="center">
| <a href="https://arxiv.org/pdf/"><b>Paper (Jakiro)</b></a> | 
</p>

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

##

<p align="center">
  <img src="./figs/A100-40G_MTbench_S_Comparison_Temperature_1.png" alt="benchmark" width="790">
</p>

**Jakiro** is an advanced approach designed to enhance speculative decoding (SD) for large language models. By integrating Mixture of Experts (MoE), Jakiro enables independent experts to generate diverse predictions, effectively decoupling correlations among candidates and addressing a key limitation of traditional tree-based sampling. Jakiro significantly boosts prediction accuracy and inference speed, setting a new state-of-the-art (SOTA) in speculative decoding. Extensive experiments across various models demonstrate its robustness and effectiveness in real-world applications.
