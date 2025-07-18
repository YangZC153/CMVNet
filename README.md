# CMVNet: Centerline-Guided Multi-View Network for Continuous Mandibular Canal Segmentation

by Zhencheng Yang, Yuchen Zheng, Yan Yang, Min Tan, Jiajun Ding, and Fan Yang.

### Introduction

This repository is for our paper 'CMVNet: Centerline-Guided Multi-View Network for Continuous Mandibular Canal Segmentation'. 

**Note: Our dataset and code are currently being refined and will be available soon.**

### Train and Test
1. Write the full path of the CBCT data in the `configs/CMV_config`.
2. Run the model: `python main_CMV.py`.

### Dataset - ContMC

We introduce the ContMC dataset with 100 CBCT cases emphasizing anatomical continuity, annotated by 5 experienced dentists from Zhejiang Provincial People's Hospital.

### Data Registration

We have released the CBCT data and GT annotation of ContMC. If you want to apply for these data, please complete the registration form in following link (XXXXXXXXX), and then send to Yuchen Zheng (XXXXXXXXX). He will send you the download link when recieve the data registration form.

### Acknowledgements

This code is inspired by the AImageLab alveolar_canal. We thank Marco Cipriano et al. for their work and their publicly available implementation.

### Citation

If the code or data is useful for your research, please consider citing:

```bibtex
@article{XXXXXXX,
    title={CMVNet: Centerline-Guided Multi-View Network for Continuous Mandibular Canal Segmentation},
    author={Yang, Zhencheng and Zheng, Yuchen and Yang, Yan and Tan, Min and Ding, Jiajun and Yang, Fan},
    journal={XXXXXX},
    volume={XX},
    number={XX},
    pages={XX--XX},
    year={2025},
    publisher={XXXX}
}
