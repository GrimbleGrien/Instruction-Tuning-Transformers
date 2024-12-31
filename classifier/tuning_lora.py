import argparse
import os
import warnings
import pandas as pd
import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a seq2seq model with LoRA.")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name.")
    parser.add_argument("--text_column", type=str, default="sentence", help="Text column in the dataset.")
    parser.add_argument("--label_column", type=str, default="text_label", help="Label column in the dataset.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--output_path", type=str, required=True, help="Path ending in .txt to save the output predictions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing train and test CSV files.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    train_csv_path = os.path.join(args.output_dir, "train_instruct.csv")
    test_csv_path = os.path.join(args.output_dir, "test_instruct.csv")

    train_dataset = CustomDataset(csv_path=train_csv_path, root_dir=args.output_dir)
    test_dataset = CustomDataset(csv_path=test_csv_path, root_dir=args.output_dir)

    train_dataset = Dataset.from_dict({
        args.text_column: list(train_dataset.sentences),
        args.label_column: list(train_dataset.text_labels)
    })
    test_dataset = Dataset.from_dict({
        args.text_column: list(test_dataset.sentences),
        args.label_column: list(test_dataset.text_labels)
    })

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir='/')

    def preprocess_function(examples):
        inputs = examples[args.text_column]
        targets = examples[args.label_column]
        model_inputs = tokenizer(inputs, max_length=args.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=30, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets_train = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=[args.text_column, args.label_column],
        load_from_cache_file=False,
        desc="Running tokenizer on training dataset",
    )

    processed_datasets_test = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=[args.text_column, args.label_column],
        load_from_cache_file=False,
        desc="Running tokenizer on test dataset",
    )

    train_dataloader = DataLoader(
        processed_datasets_train,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        processed_datasets_test,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    lora_config = LoraConfig(
        r=2,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir='/', device_map="auto")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        auto_find_batch_size=True,
        learning_rate=5e-4,
        num_train_epochs=1,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets_train,
        eval_dataset=processed_datasets_test,
        tokenizer=tokenizer,
    )
    model.config.use_cache = False
    trainer.train()

    peft_model_id = os.path.join(args.output_dir, 'lora-flan-t5-base')
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    model.eval()
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

    with open(args.output_path, 'w') as f1:
        for pred, true in zip(eval_preds, test_dataset[args.label_column]):
            f1.write(f"True: {true} Pred: {pred}\n")

class CustomDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.sentences = self.df['input']
        self.text_labels = self.df['target']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        text_labels = self.text_labels[idx]
        sample = {'sentence': sentence, 'text_labels': text_labels}
        return sample

if __name__ == "__main__":
    main()
