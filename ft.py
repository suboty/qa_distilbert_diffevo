import copy
import os
import warnings

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from transformers import AdamW, default_data_collator

from src.data import DataQA
from src.utils import Utils
from src.fine_tuning import train as ft_train
from src.fine_tuning import set_seed
from logger import logger


ITERATIONS = os.environ.get('ITERATIONS') if os.environ.get('ITERATIONS') else 20

if __name__ == '__main__':
    warnings.simplefilter("ignore")

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Available device: {device}')

    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    utils = Utils(tokenizer)

    root_dataset = load_dataset("squad")

    for _iter in range(ITERATIONS):

        if _iter == 10:
            set_seed(424)

        if _iter % 10 == 0 and _iter != 0:
            train_sample_length = ((_iter - 10) + 1) * 100
        else:
            train_sample_length = (_iter + 1) * 100

        val_sample_length = 500

        print(f'\n{"#"*30}\nITERATION {_iter+1}: {train_sample_length}-{val_sample_length}\n{"#"*30}\n')
        logger.info(f'\n{"#"*30}\nITERATION {_iter+1}: {train_sample_length}-{val_sample_length}\n{"#"*30}\n')

        dataset = copy.deepcopy(root_dataset)

        dataset['train'] = dataset['train'].select([i for i in range(train_sample_length)])
        dataset['validation'] = dataset['validation'].select([i for i in range(val_sample_length)])

        train_dataset = DataQA(dataset, mode="train", utils=utils)
        val_dataset = DataQA(dataset, mode="validation", utils=utils)
        eval_dataloader = DataLoader(
            val_dataset, collate_fn=default_data_collator, batch_size=2
        )

        validation_processed_dataset = dataset["validation"].map(utils.preprocess_validation_examples,
                                                                 batched=True,
                                                                 remove_columns=dataset["validation"].column_names, )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=2,
        )

        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        ft_train(model=model,
                 optimizer=optimizer,
                 tokenizer=tokenizer,
                 train_dataloader=train_dataloader,
                 eval_dataloader=eval_dataloader,
                 dataset=dataset,
                 validation_processed_dataset=validation_processed_dataset,
                 epochs=10,
                 device=device)
