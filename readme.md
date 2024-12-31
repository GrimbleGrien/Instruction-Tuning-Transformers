# Instruction Tuning Transformers for Text Classification
This repository provides a pipeline for instruction-tuning Transformer models and performing label matching for text classification tasks described in the [paper](https://arxiv.org/abs/2404.01669) on [CAVES](https://arxiv.org/abs/2204.13746) Dataset. 

The workflow consists of data preparation, instruction tuning using LoRA (Low-Rank Adaptation), and matching text predictions to tailor-made labels.

## Data Preparation
```bash
python data_preparation.py \
    --train train.csv \
    --eval eval.csv \
    --test test.csv \
    --out_dir dataset_instruct
```

## Instruction Tuning (LoRA)
```bash
python tuning_lora.py \
    --model_name "google/flan-t5-base" \
    --tokenizer_name "google/flan-t5-base" \
    --text_column "sentence" \
    --label_column "text_label" \
    --max_length 128 \
    --batch_size 8 \
    --output_path "./output.txt" \
    --output_dir "./data"
```

## Label Matching
```bash
python label_matcher.py \
    --model "all-MiniLM-L6-v2" \
    --output_path "./lora_prediction_flan_t5_base.txt"
    --test_dataset "dataset_instruct/test_instruct.csv"
```

Stay tuned for more customizations and guides. In the meantime feel free to clone this repo in colab and experiment yourself :)

<a target="_blank" href="https://colab.research.google.com/github/GrimbleGrien/Instruction-Tuning-Transformers/blob/main/instruction_tuning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
