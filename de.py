import copy
import warnings

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from transformers import AdamW, default_data_collator

from src.data import DataQA
from src.utils import Utils
from src.fine_tuning import set_seed
from src.differential_evolution import train as de_train


ITERATION = 10

if __name__ == '__main__':
    # TODO: fix old logic
    warnings.simplefilter("ignore")

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Available device: {device}')

    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    utils = Utils(tokenizer)

    root_dataset = load_dataset("squad")

    for _iter in range(ITERATION):
        dataset = copy.deepcopy(root_dataset)
        train_sample_length = (_iter + 1) * 1000
        val_sample_length = (_iter + 1) * 100
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

        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        de_train(model=model,
                 train_dataloader=train_dataloader,
                 eval_dataloader=eval_dataloader,
                 validation_processed_dataset=validation_processed_dataset,
                 dataset=dataset,
                 all_mode=False,
                 device=device,
                 number_of_samples=-1,
                 number_of_iterations=10)
