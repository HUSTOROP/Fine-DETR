# Fine-Grained PCB Defect Detection Transformer Based on Feature Enhancement with Defect-Free Samples

This repository is the official implementation of Fine-DETR, which is built upon the RTDETR codebase. Fine-DETR focuses on enhancing defect detection performance by mining pattern information from defect-free samples, addressing the limitation that traditional methods ignore the value of defect-free samples.

## 📌 Main Contributions & Methods
### Core Modules of Fine-DETR
1. Semantic Background Suppression (SBS)
SBS separates defect features and defect-free features, and adopts different learning strategies for each, ensuring the stable learning of the model on defect-related features.
2. Intersection Box Guided Perception (IBGP)
IBGP strengthens the learning of inconspicuous features, further enhancing the model's ability to learn defect features.

### Extended Implemented Modules
To further optimize detection performance, the repository also implements the following modules:
- Loss functions: CIoU, EIoU, alpha-IoU
- Detection head: cosine detection head
- Attention mechanism: CBAM residual attention mechanism
- Downsampling method: Adown efficient downsampling method

## 🙏 Acknowledgements
This repository is built based on the RTDETR codebase. We sincerely thank the authors of RTDETR for their open-source contribution.
