# ReMask-CC: RemoteCLIP-Guided Semantic Change Masking for Remote Sensing Image Change Captioning

**Published in IEEE Geoscience and Remote Sensing Letters (GRSL), 2026.**

[IEEE Xplore](https://ieeexplore.ieee.org/document/11733685) · [DOI](https://doi.org/10.1109/LGRS.2026.3742178) · [MOSAIC Research Group](https://avesis.yildiz.edu.tr/arastirma-grubu/mosaic)

[![GitHub stars](https://img.shields.io/github/stars/ChangeCapsInRS/ReMaskCC?style=social)](https://github.com/ChangeCapsInRS/ReMaskCC)

## 🔎 Summary

ReMask-CC describes what changed between two remote sensing images of the same scene. It uses a frozen RemoteCLIP ViT-L/14 backbone to extract mid- and high-level patch features from both dates. At each level, patch-wise cosine dissimilarity produces a change mask that downweights unchanged regions. The model fuses the masked features and uses a Transformer decoder to generate a caption. A lightweight residual FeatureCNN refines local features before masking, and a dynamic triplet loss improves cross-temporal feature discrimination during training.

## 📘 Datasets

The paper evaluates ReMask-CC on **LEVIR-CC**, **DUBAI-CC**, and **WHU-CDC**. The commands below use LEVIR-CC, the default dataset in the training and evaluation scripts.

## ⚙️ Installation and Dependencies

The training and evaluation scripts use an NVIDIA GPU with CUDA. The following setup uses Python 3.10 and PyTorch 2.4.1 with CUDA 12.1. Run the remaining commands from the repository root.

```bash
git clone https://github.com/ChangeCapsInRS/ReMaskCC.git
cd ReMaskCC
conda create -n remaskcc python=3.10 -y
conda activate remaskcc
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" Pillow "opencv-python<4.12" tqdm einops openai-clip==1.0.1
```

For a different CUDA version, see the [PyTorch installation matrix](https://docs.pytorch.org/get-started/previous-versions/#v241).

Download [`RemoteCLIP-ViT-L-14.pt`](https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-L-14.pt) and place it in the repository root. This file contains the backbone weights. Evaluation also requires a trained ReMask-CC checkpoint.

## 📊 Evaluation Metrics

The evaluation script reports **BLEU-1 through BLEU-4**, **METEOR**, **ROUGE-L**, and **CIDEr**. It also reports the mean of BLEU-4, METEOR, ROUGE-L, and CIDEr. The training script uses this mean on the validation set to select the best checkpoint.

Download the `eval_func` folder and place it in the repository root alongside `train.py` and `test.py`.

[Download the evaluation scripts (Google Drive)](https://drive.google.com/file/d/1opsJk5yfA1T-FblYySbaIPs5q3VaWa4V/view?usp=drive_link)

```text
eval_func/
├── bleu/
├── cider/
├── meteor/
└── rouge/
```

For METEOR, make sure Java is available on your system path and keep the METEOR JAR in the location expected by `eval_func/meteor/meteor.py`.

## 📂 Data Preparation

### Option 1. Download the Prepared LEVIR-CC Folder

[Download the prepared LEVIR-CC folder (Google Drive)](https://drive.google.com/file/d/1HQNCnFJMC1VoBHFU105PZfdkCx655vxH/view?usp=drive_link)

Place the downloaded folder at `data/LEVIR_CC/`. It contains the images, caption tokens, vocabulary, and split lists. You can proceed directly to training or evaluation.

```text
data/LEVIR_CC/
├── images/
├── tokens/
├── vocab.json
├── train.txt
├── val.txt
└── test.txt
```

### Option 2. Reproduce the Preprocessing

Download the image archive and caption annotations from the [LEVIR-CC repository](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset). Keep the original split and A/B directories when extracting the images.

```text
data/LEVIR_CC/
├── LevirCCcaptions.json
└── images/
    ├── train/
    │   ├── A/
    │   └── B/
    ├── val/
    │   ├── A/
    │   └── B/
    └── test/
        ├── A/
        └── B/
```

Run the preprocessing script from the repository root.

```bash
python preprocess_data.py --dataset LEVIR_CC --word_count_threshold 5
```

The script writes `train.txt`, `val.txt`, `test.txt`, `tokens/`, and `vocab.json` under `data/LEVIR_CC/`.

For WHU-CDC, place the images under `data/WHU_CDC/images/` and the annotations at `data/WHU_CDC/whuCCcaptions.json`, then run preprocessing with `--dataset WHU_CDC`. For DUBAI-CC, the script reads the caption files from `data/Dubai_CC/DubaiCC/datasetDubaiCCPublic/description_jsontr_te_val/`. Place its images under `data/Dubai_CC/images/` and run preprocessing with `--dataset Dubai_CC`. Set `--data_name` to the same dataset name when training or evaluating.

## 🔎 Inference and Evaluation

[Download the pretrained LEVIR-CC checkpoint (Google Drive)](https://drive.google.com/file/d/1bxCmvwDlu83eOvkouumA0FxWgIKgm401/view?usp=drive_link). Save it as `models_checkpoint/RemoteClip14_FeatureCNN_LEVIR_CC_batchsize_32_remoteClip.pth` and evaluate it with the corresponding LEVIR-CC `vocab.json`.

To evaluate a checkpoint from your own training run, pass its filename to `--checkpoint` and its directory to `--savepath`.

```bash
python test.py --data_name LEVIR_CC --savepath ./models_checkpoint/ --checkpoint RemoteClip14_FeatureCNN_LEVIR_CC_batchsize_32_remoteClip.pth --gpu_id 0 --beam_size 7
```

The script generates captions with beam search and prints the evaluation scores. It evaluates the test split for LEVIR-CC and WHU-CDC. For DUBAI-CC, the current script uses the validation split.

## 🏋️ Training

After preparing the dataset and downloading the RemoteCLIP weights, train ReMask-CC with the following command.

```bash
python train.py --data_name LEVIR_CC --network remoteClip --mask Self --featureCNN True --featureCNNLayer 3 --rankingLossCo 0.2 --train_batchsize 32 --num_epochs 50 --gpu_id 0 --model_name ReMaskCC
```

The best validation checkpoint is saved to `./models_checkpoint/ReMaskCC_LEVIR_CC_batchsize_32_remoteClip.pth`.

## 📑 Citation

If you use ReMask-CC in your research, please cite our paper.

```bibtex
@article{karaca2026remaskcc,
  author  = {Karaca, Ali Can and Oruk, Meryem and Yilmaz, Bedirhan},
  title   = {{ReMask-CC}: {RemoteCLIP}-Guided Semantic Change Masking for Remote Sensing Image Change Captioning},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  volume  = {23},
  doi     = {10.1109/LGRS.2026.3742178}
}
```

## 🙏 References

- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) provides the pretrained vision backbone.
- [RSICCformer and LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC) provide a reference implementation and benchmark for remote sensing change captioning.
