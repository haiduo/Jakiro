<div align="center">
  <div style="display: inline-block; vertical-align: middle;">
    <img src="figs/jakiro.webp" alt="Jakiro" width="220">
  </div>
  <div style="display: inline-block; vertical-align: middle; margin-left: 10px;">
    <h1>Jakiro</h1>
  </div>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2502.06282"><b>Paper (Jakiro)</b></a>
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
The following shows the actual measured inference speeds of Jakiro and EAGLE-2 on a single RTX 4090 GPU with 24GB of memory using the Vicuna 7B model. As shown, Jakiro has a faster decoding speed and a higher compression ratio.

<table align="center" style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 1px;"><img src="./figs/Eagle2.gif" alt="EAGLE-2 Demo" style="max-width: 46.5%; height: auto;"></td>
    <td style="padding-left: 1px;"><img src="./figs/Jakiro.gif" alt="Jakiro Demo" style="max-width: 46.5%; height: auto;"></td>
  </tr>
</table>

## Update
**2025.3.24:** Jakiro and Jakiro* (codes and checkpoints) is released.


## Todo
- [ ] Support more LLMs such as Mixtral 8x7B, Qwen-2, and DeepSeek-R1.
- [ ] Jakiro-V2 for Multimodal Large Language Models (LLaVA and InstructBLIP).


## Setup & Installation
```bash
git clone git@github.com:haiduo/Jakiro.git
cd Jakiro
pip install -r requirements.txt
```

## Jakiro Weights
 Base Model      | Jakiro                                                                                            | \# Parameters | Jakiro*             | \# Parameters 
:---------------:|:-------------------------------------------------------------------------------------------------:|:-------------:|:-------------------:|:-------------:
 Vicuna-7B-v1.3  | [Jakiro-Vicuna-7B-v1.3](https://drive.google.com/drive/folders/1HBHMaXtvh4dFYEFuQASiJAb27sTREeGJ?usp=sharing)     | 0.38B         | [Jakiro*-Vicuna-7B-v1.3](https://drive.google.com/drive/folders/1TeyQ9f8TsvXr0lj_BysTcaAvpKIRGhgU?usp=sharing)    | 0.23B         
 Vicuna-13B-v1.3 | [Jakiro-Vicuna-13B-v1.3](https://drive.google.com/drive/folders/197pC42RU92r9tMW4tv1q4c-cJCigZk00?usp=sharing)    | 0.53B         | [Jakiro*-Vicuna-13B-v1.3](https://drive.google.com/drive/folders/1BYl-VU2xq_zMMJPwmCmsX3dFV-hF7_Zg?usp=sharing)   | 0.35B         
 LLaMA2-Chat 7B  | [Jakiro-LLaMA2-Chat-7B](https://drive.google.com/drive/folders/1FS7j8V6Lnx1xuJJg03VQZwwBAxhHosW2?usp=sharing)     | 0.38B         | [Jakiro*-LLaMA2-Chat-7B](https://drive.google.com/drive/folders/1_OwoUJOsMqInXN8FPUXZE0lZcYXXwZzz?usp=sharing)    | 0.23B         
 LLaMA2-Chat 13B | [Jakiro-LLaMA2-Chat-13B](https://drive.google.com/drive/folders/1bD2H6Yl6Uy5WR6VvODFMZBBJP02W9oe4?usp=sharing)    | 0.53B         | [Jakiro*-LLaMA2-Chat-13B](https://drive.google.com/drive/folders/1EFGrkuE3jzFbHAxzCVJCvBm4DBrsCa-b?usp=sharing)   | 0.35B         

Notably, other Jakiro LLMs' checkpoints are being organized and will be uploaded soon. If you need them urgently, please email the author for permission.


## Inference
The inference code we provide automatically distributes model weights across multiple GPUs, enabling you to run models that exceed the memory capacity of a single GPU.


# Train

### Generate Train Data
You can run the following command to generate the training data.
```bash
python -m jakiro.ge_data.allocation --outdir [path of data]
```
### Train the Auto-regression Head
#### For Jakiro:
```bash
# Switch to training mode. In jakiro.model.cnets.py, you need to uncomment self.mlp_moe = MixtralSparseMoeBlock_train(config) and comment out self.mlp_moe = MixtralSparseMoeBlock(config)
accelerate launch -m --mixed_precision=bf16 jakiro.train.main --tmpdir [path of data] --cpdir [path of checkpoints] --configpath [path of config file]
```
#### For Jakiro*:
```bash
# Switch to training mode. In jakiro_star.model.cnets.py, you need to uncomment self.mlp_moe = MixtralSparseMoeBlock_train(config) and comment out self.mlp_moe = MixtralSparseMoeBlock(config)
accelerate launch -m --mixed_precision=bf16 jakiro_star.train.main --tmpdir [path of data] --cpdir [path of checkpoints] --configpath [path of config file]
```

Notably, *jakiro/train* and *jakiro_star/train* provides examples of configuration files. 

You can also use DeepSpeed for training.

```bash
cd jakiro/train or jakiro_star/train
deepspeed main_deepspeed.py --deepspeed_config ds_config.json
```


## Evaluation
You can test the speed of EAGLE on MT-bench using the following command.  
#### For Jakiro:
```bash
# Switch to training mode. In jakiro.model.cnets.py, you need to uncomment self.mlp_moe = MixtralSparseMoeBlock_train(config) and comment out self.mlp_moe = MixtralSparseMoeBlock(config)
python -m jakiro.evaluation.gen_ea_answer_vicuna(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of jakiro weight]\ 
		 --base-model-path [path of the original model]\
```
#### For Jakiro*:
```bash
# Switch to training mode. In jakiro_star.model.cnets.py, you need to uncomment self.mlp_moe = MixtralSparseMoeBlock_train(config) and comment out self.mlp_moe = MixtralSparseMoeBlock(config)
python -m jakiro.evaluation.gen_ea_answer_vicuna(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of jakiro weight]\ 
		 --base-model-path [path of the original model]\
```

If you need specific acceleration ratios, you will also need to run the following command to get the speed of vanilla auto-regression.
```bash
python -m jakiro.evaluation.gen_baseline_answer_vicuna(or jakiro_star.evaluation.gen_baseline_answer_vicuna)\
		 --ea-model-path [path of jakiro weight]\ 
		 --base-model-path [path of the original model]\
```
The above two commands will each generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the ratio of speeds.

Of course, you can also batch test and collect statistics of Jakiro-LLMs using the following script:
```bash
bash eval_metric\eval_vicuna_Jakiro_all.sh (or eval_metric\eval_vicuna_Jakiro-star_all.sh)
```


### With UI
We have provided a suggested web interface, which you can use by running the following command. After the model is fully loaded, a URL will be output in the terminal, which you can enter into your browser to access.
```bash
python -m jakiro.application.webui --ea-model-path [path of Jakiro weight]\ 
		--base-model-path [path of the original model]\
		--model-type [vicuna\llama2\llama3]\
        --total-token [int]
```
The *total-token* is the number of draft tokens. For smaller models and advanced GPUs, this value can be set larger. Adjusting according to the specific device and model can achieve better results. If set to -1, Jakiro will automatically configure this parameter.


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
