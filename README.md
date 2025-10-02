---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{}
---

## 🏠 About of Our Dataset

<!-- Provide a quick summary of the dataset. -->

This dataset is provided for the research of 3D visual decoding from EEG signals. It comprises multimodal analysis data and comprehensive EEG recordings collected from 12 participants exposed to 72 categories of 3D visual stimuli. The dataset originates from our [research paper](https://arxiv.org/abs/2411.12248), which has been accepted by <b>CVPR2025</b>. Detailed usage instructions and technical documentation can be accessed through [GitHub](https://github.com/gzq17/neuro-3D).

##  🔍 Overview

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

- **EEGdata:** Collected EEG signals, including responses to dynamic video stimuli and static image stimuli. All data have been preprocessed.
- **image.zip:** Static image stimulus.
- **video_new.zip:** Dynamic video stimulus.
- **point_cloud.zip:** Corresponding point cloud data.
- **point_cloud_simple.zip:** Simplified color point cloud data.
- **model3d.zip:** Original 3D object models.
- **description.json:** Textual descriptions of objects.
- **color_label.xlsx:** Color category labels of objects.
- **clip_feature.pth:** Features extracted from video sequences.

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** [GitHub](https://github.com/gzq17/neuro-3D)
- **Paper:** [arxiv](https://arxiv.org/abs/2411.12248)

## 🔗 Citation

If you find our work and this dataset helpful, please cite:
```bibtex
@article{guo2024neuro,
  title={Neuro-3D: Towards 3D Visual Decoding from EEG Signals},
  author={Guo, Zhanqiang and Wu, Jiamin and Song, Yonghao and Bu, Jiahui and Mai, Weijian and Zheng, Qihao and Ouyang, Wanli and Song, Chunfeng},
  journal={arXiv preprint arXiv:2411.12248},
  year={2024}
}
```

## 📧 Contact us

If you have any questions about this dataset, please do not hesitate to contact me.

Zhanqiang Guo: guozq21@mails.tsinghua.edu.cn
