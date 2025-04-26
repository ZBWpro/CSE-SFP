# Read Me

CSE-SFP: Enabling Unsupervised Sentence Representation Learning via a Single Forward Pass

**This paper has been accepted to SIGIR 2025. (Full)**

***

## Quick Start

- Python Version: 3.9.18

- Install Dependencies

  ```bash
  cd code
  pip install -r requirements.txt
  ```

- Download Data

  ```bash
  cd data
  bash download_wiki.sh
  ```
  
- Download SentEval

  ```bash
  cd SentEval/data/downstream/
  bash download_dataset.sh
  ```

- Train CSE-SFP

  ```bash
  cd code
  nohup torchrun --nproc_per_node=4 train.py > nohup.log & # 4090 * 4
  ```

****

## Checkpoints

- Link: https://drive.google.com/drive/folders/1IJEq_1F2X8uv_zyj0TgHM_LFOopA4YaW?usp=sharing

## Acknowledgement

- Our code is based on PromptEOL

## Related Work

- Github: [Pcc-tuning](https://github.com/ZBWpro/Pcc-tuning)

  Paper: [Pcc-tuning: Breaking the Contrastive Learning Ceiling in Semantic Textual Similarity](https://arxiv.org/abs/2406.09790)

  Conference: :star2: EMNLP 2024, **Main**

- Github: [STS-Regression](https://github.com/ZBWpro/STS-Regression)

  Paper: [Advancing Semantic Textual Similarity Modeling: A Regression Framework with Translated ReLU and Smooth K2 Loss](https://arxiv.org/abs/2406.05326)

  Conference: :star2: EMNLP 2024, **Main**

- Github: [CoT-BERT](https://github.com/ZBWpro/CoT-BERT)

  Paper: [CoT-BERT: Enhancing Unsupervised Sentence Representation through Chain-of-Thought](https://arxiv.org/abs/2309.11143) 

  Conference: :star2: ICANN 2024, **Oral**

- Github: [PretCoTandKE](https://github.com/ZBWpro/PretCoTandKE)

  Paper: [Simple Techniques for Enhancing Sentence Embeddings in Generative Language Models](https://arxiv.org/abs/2404.03921)​ 

  Conference: :star2: ICIC 2024, **Oral**
