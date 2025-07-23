<p align="center">
  <img src="assets/logo.png" alt="Logo" width="150"/>
</p>
<h3  align="center">TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation</h3>
<p align="center">
  <strong>Mohan Xu<sup>*</sup>, Kai Li<sup>*</sup>, Guo Chen, Xiaolin Hu</strong><br>
    <strong>Tsinghua University, Beijing, China</strong><br>
    <strong><sup>*</sup>Equal contribution</strong><br>
  <a href="https://arxiv.org/abs/2410.01469">📜 ICLR 2025</a> | <a href="https://cslikai.cn/TIGER/">🎶 Demo</a> | <a href="https://huggingface.co/datasets/JusperLee/EchoSet">🤗 Dataset</a>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.TIGER" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/JusperLee/TIGER?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
</p>

<p align="center">

> TIGER is a lightweight model for speech separation which effectively extracts key acoustic features through frequency band-split, multi-scale and full-frequency-frame modeling.

## 💥 News

- **[2025-01-23]** We release the code and pre-trained model of TIGER! 🚀
- **[2025-01-23]** We release the TIGER model and the EchoSet dataset! 🚀

## 📜 Abstract

In this paper, we propose a speech separation model with significantly reduced parameter size and computational cost: Time-Frequency Interleaved Gain Extraction and Reconstruction Network (TIGER). TIGER leverages prior knowledge to divide frequency bands and applies compression on frequency information. We employ a multi-scale selective attention (MSA) module to extract contextual features, while introducing a full-frequency-frame attention (F^3A) module to capture both temporal and frequency contextual information. Additionally, to more realistically evaluate the performance of speech separation models in complex acoustic environments, we introduce a novel dataset called EchoSet. This dataset includes noise and more realistic reverberation (e.g., considering object occlusions and material properties), with speech from two speakers overlapping at random proportions. Experimental results demonstrated that TIGER significantly outperformed state-of-the-art (SOTA) model TF-GridNet on the EchoSet dataset in both inference speed and separation quality, while reducing the number of parameters by 94.3% and the MACs by 95.3%. These results indicate that by utilizing frequency band-split and interleaved modeling structures, TIGER achieves a substantial reduction in parameters and computational costs while maintaining high performance. Notably, TIGER is the first speech separation model with fewer than 1 million parameters that achieves performance close to the SOTA model.

## TIGER

Overall pipeline of the model architecture of TIGER and its modules.

![TIGER Model Architecture](assets/TIGER.png)

## Results

Performance comparisons of TIGER and other existing separation models on ***Libri2Mix, LRS2-2Mix, and EchoSet***. Bold indicates optimal performance, and italics indicate suboptimal performance.

![TIGER Model Architecture](assets/result.png)

Efficiency comparisons of TIGER and other models.

![TIGER Model Architecture](assets/efficiency.png)

Comparison of performance and efficiency of cinematic sound separation models on DnR. '*' means the result comes from the original paper of DnR.

![TIGER Model Architecture](assets/dnr.png)

## 📦 Installation

```bash
git clone https://github.com/JusperLee/TIGER.git
cd TIGER
pip install -r requirements.txt
```

## 🚀 Quick Start

### Test with Pre-trained Model

```bash
# Test using speech
python inference_speech.py --audio_path test/mix.wav

# Test using DnR
python inference_dnr.py --audio_path test/test_mixture_466.wav
```

### Train with EchoSet

```bash
python audio_train.py --conf_dir configs/tiger.yml
```

### Evaluate with EchoSet

```bash
python audio_test.py --conf_dir configs/tiger.yml
```

## 📖 Citation

```bibtex
@article{xu2024tiger,
  title={TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation},
  author={Xu, Mohan and Li, Kai and Chen, Guo and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2410.01469},
  year={2024}
}
```

## 📧 Contact

If you have any questions, please feel free to contact us via `tsinghua.kaili@gmail.com`.
