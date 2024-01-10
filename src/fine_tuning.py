import random,time
import numpy as np
import torch

from src.utils import Utils, format_time, predict_answers_and_evaluate


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model,
          optimizer,
          tokenizer,
          train_dataloader,
          eval_dataloader,
          dataset,
          validation_processed_dataset,
          epochs=10,
          device='cuda',
          verbose=False):
    stats = []

    total_train_time_start = time.time()

    for epoch in range(epochs):
        if verbose:
            print(' ')
            print(f'=====Epoch {epoch + 1}=====')
            print('Training....')

        t0 = time.time()

        training_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed_time = format_time(time.time() - t0)
                if verbose:
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step,
                                                                                len(train_dataloader),
                                                                                elapsed_time))

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            model.zero_grad()

            result = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           start_positions=start_positions,
                           end_positions=end_positions,
                           return_dict=True)

            loss = result.loss

            training_loss += loss.item()

            loss.backward()

            optimizer.step()

        avg_train_loss = training_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        if verbose:
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            print("")
            print("Running Validation...")

        t0 = time.time()

    model.eval()

    start_logits, end_logits = [], []
    for step, batch in enumerate(eval_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            result = model(input_ids=input_ids,
                            attention_mask=attention_mask, return_dict=True)

        start_logits.append(result.start_logits.cpu().numpy())
        end_logits.append(result.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    answers, metrics_ = predict_answers_and_evaluate(start_logits, end_logits,
                                                     validation_processed_dataset,
                                                     dataset["validation"])
    print(f'### Exact match: {metrics_["exact_match"]}, F1 score: {metrics_["f1"]}')

    validation_time = format_time(time.time() - t0)

    if verbose:
        print("--- Validation took: {:}".format(validation_time))
        print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_train_time_start)))
