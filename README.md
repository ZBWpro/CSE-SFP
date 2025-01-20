# Read Me

CSE-SFP: Enabling Unsupervised Sentence Representation Learning via a Single Forward Pass

***

## Checkpoints

- Link: https://drive.google.com/drive/folders/1IJEq_1F2X8uv_zyj0TgHM_LFOopA4YaW?usp=sharing

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
