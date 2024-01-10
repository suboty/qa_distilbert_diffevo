import copy
import warnings

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from transformers import AdamW, default_data_collator

from src.data import DataQA
from src.utils import Utils
from src.fine_tuning import train as ft_train
from src.differential_evolution import train as de_train


ITERATION = 1

if __name__ == '__main__':
    warnings.simplefilter("ignore")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Available device: {device}')

    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    utils = Utils(tokenizer)

    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    root_dataset = load_dataset("squad")

    for _iter in range(ITERATION):
        dataset = copy.deepcopy(root_dataset)
        train_sample_length = _iter * 1000
        val_sample_length = _iter * 100
        dataset['train'] = dataset['train'].select([i for i in range(train_sample_length)])
        dataset['validation'] = dataset['validation'].select([i for i in range(val_sample_length)])

        train_dataset = DataQA(dataset, mode="train", utils=utils)
        val_dataset = DataQA(dataset, mode="validation", utils=utils)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=2,
        )
        eval_dataloader = DataLoader(
            val_dataset, collate_fn=default_data_collator, batch_size=2
        )

        validation_processed_dataset = dataset["validation"].map(utils.preprocess_validation_examples,
                                                                 batched=True,
                                                                 remove_columns=dataset["validation"].column_names, )

        ft_train(model,
                 optimizer,
                 tokenizer,
                 train_dataloader,
                 eval_dataloader,
                 dataset,
                 validation_processed_dataset,
                 epochs=1,
                 device=device)

        de_train(model,
                 eval_dataloader,
                 validation_processed_dataset,
                 dataset,
                 all_mode=False,
                 device=device,
                 number_of_samples=-1,
                 number_of_iterations=1)
