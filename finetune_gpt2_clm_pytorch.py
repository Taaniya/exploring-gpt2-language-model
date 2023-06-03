# -*- coding: utf-8 -*-
"""
This pipeline fine-tunes GPT2 family models on causal language modelling task.
This script assumes a CSV input dataset, with training data present in 'text' column while optionally labels (if using for text classification) 
included in the 'label' column. Modify the corresponding lines in the script to suit your dataset.
"""

import os
import argparse
import math
import random
import numpy as np
import torch

import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

def _concat_input_target(row):
    """
    Concatenates input sentence / promt and target labels when using GPT2 for text classification task.
    """
    row['text'] = "###Text: " + row['text'] + " ###Label:" + row['label'].strip()
    return row

def main():
    parser = argparse.ArgumentParser(
        description="Finetuning Decoder only Transformer model with causal language modelling objective", add_help=True)
    parser.add_argument("--model-name", type=str, default="gpt2",
                        help="pre-trained model name or path")
    parser.add_argument("--seed", type=int, default=42, help="random seed value")
    parser.add_argument("--batch-size", type=int, required=True, help="batch size for training")
    parser.add_argument("--epoch", type=int, default=1, help="No. of training epochs")
    parser.add_argument("--model-dir", type=str, required=True, help="path to directory to save model")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay rate")
    parser.add_argument("--model-version-no", type=int)
    parser.add_argument("--dataset-path", type=str, required=True, help="path to load dataset from")
    args = parser.parse_args()

    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    dataset_path = args.dataset_path
    learning_rate = args.lr
    num_epochs = args.epoch
    model_dir = args.model_dir
    batch_size = args.batch_size
    model_version = args.model_version_no
    model_checkpoint = args.model_name
    weight_decay = args.weight_decay
    model_save_path = f"{model_dir}/{model_checkpoint}-{model_version}"

    # load training data
    if os.path.isfile(dataset_path):
        datasets = load_dataset("csv", data_files={"train": [dataset_path]}, download_mode='force_redownload')
        print(f"dataset loaded: {datasets} from path:{dataset_path}")

    else:
        print("invalid file path.")

    # Preprocess for text classification. Call function to concatenate input sentence and target labels. Print 1st 5 lines.
    # Uncomment below lines for text classification task
    # preprocessed_dataset = datasets.map(_concat_input_target)
    # print(f"preprocessed dataset: {preprocessed_dataset['train']['text'][:5]}")
    
    # Load Tokenizer. Use fast implementation of tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # postpone padding in preprocessing step to apply dynamic padding later
    # For text classification task with more columns in the dataset, remove all other columns in below step along with 'text' column
    tokenized_datasets = dataset.map(lambda example: tokenizer(example['text']), batched=True, num_proc=4,
                                     remove_columns=["text"])
    # set EOS as pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Perform dynamic padding with Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    print("Model loaded.")

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # The Trainer performs training data shuffling during training in every epoch as it used torch's DistributedSampler
    # class to sample data
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L862
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train']
    )
    trainer.train()

    print("Training complete!")
    print("saving model to {}".format(model_save_path))

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()
