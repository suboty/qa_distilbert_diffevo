import time
import random

import torch
import numpy as np

from src.utils import format_time, predict_answers_and_evaluate


class DE:
    def __init__(self,
                 all_mode=False,
                 device='cpu',
                 number_of_samples=-1,
                 number_of_iterations=10):
        self.all_mode = all_mode
        self.device = device
        self.number_of_samples = number_of_samples
        self.number_of_iterations = number_of_iterations
        self.losses = {}

    def get_states(self, model, list_mode=False):
        if not list_mode:
            states = {}
            for key in model.state_dict().keys():
                if self.all_mode:
                    states[key] = model.state_dict()[key]
                else:
                    if 'qa_outputs' not in key:
                        continue
                    else:
                        states[key] = model.state_dict()[key]
            return states
        else:
            states = []
            for key in model.state_dict().keys():
                if self.all_mode:
                    states.append((model.state_dict()[key].cpu().numpy()))
                else:
                    if 'qa_outputs' not in key:
                        continue
                    else:
                        states.append((model.state_dict()[key].cpu().numpy()))
            return states

    def bert_fobj(self, train_dataloader, model, states, _popsize):
        for i, key in enumerate(x for x in list(model.state_dict().keys()) if 'qa_outputs' in x):
            if not self.all_mode:
                if 'qa_outputs' in key:
                    pass
                    model.state_dict()[key] = states[i]
            else:
                model.state_dict()[key] = states[i]

        results = []
        data = list(train_dataloader)

        if self.number_of_samples != -1:
            for i in range(self.number_of_samples):
                a = random.choice(data)

                input_ids = a['input_ids'].to(self.device)
                attention_mask = a['attention_mask'].to(self.device)
                start_positions = a['start_positions'].to(self.device)
                end_positions = a['end_positions'].to(self.device)

                _result = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions,
                                return_dict=True)
        else:
            for _data in data:
                input_ids = _data['input_ids'].to(self.device)
                attention_mask = _data['attention_mask'].to(self.device)
                start_positions = _data['start_positions'].to(self.device)
                end_positions = _data['end_positions'].to(self.device)

                _result = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions,
                                return_dict=True)

                results.append(_result.loss.cpu().detach().numpy())

        _avg_loss = sum(results) / len(results)
        self.losses[_popsize].append(_avg_loss)

        print(f'--- individual {_popsize} gets loss {_avg_loss}')

        return _avg_loss

    def de(self, fobj, bounds, model,
           mut=0.8,
           crossp=0.7,
           popsize=5):
        its = self.number_of_iterations
        model.eval().to(self.device)
        states = self.get_states(model, list_mode=True)
        dimensions = [layer.shape for layer in states]
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)

        print('Population 0')

        pop = []
        for _popsize in range(popsize):
            self.losses[_popsize] = []
            _pop = []
            for state in states:
                _pop.append(min_b + np.random.rand(*state.shape) * diff)
            pop.append(_pop)

        fitness = np.asarray([fobj(model=model, states=ind, _popsize=i) for i, ind in enumerate(pop)])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        for i_iter in range(its):
            print(f'Population {i_iter + 1}')
            for i_individual in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != i_individual]
                indexes = np.random.choice(idxs, 3, replace=False)
                a, b, c = [pop[x] for x in indexes]
                mutant = [np.clip(_a + mut * (_b - _c), 0, 1) for _a, _b, _c in zip(a, b, c)]

                cross_points = [np.random.rand(*layer.shape) < crossp for layer in states]
                if not np.any(cross_points[0]):
                    for i_layer, dimension in enumerate(dimensions):
                        cross_points[i_layer][np.random.randint(0, dimension)] = True

                trial_denorm = []
                trial = []
                for i_layer, dimension in enumerate(dimensions):
                    trial.append(np.where(cross_points[i_layer], mutant[i_layer], pop[i_individual][i_layer]))
                    trial_denorm.append(min_b + trial[i_layer] * diff)

                f = fobj(model=model, states=trial_denorm, _popsize=i_individual)
                if f < fitness[i_individual]:
                    fitness[i_individual] = f
                    pop[i_individual] = trial
                    if f < fitness[best_idx]:
                        best_idx = i_individual
                        best = trial_denorm
        yield best, fitness[best_idx]


def train(model,
          eval_dataloader,
          validation_processed_dataset,
          dataset,
          all_mode=False,
          device='cpu',
          number_of_samples=-1,
          number_of_iterations=10):
    total_train_time_start = time.time()
    de = DE(all_mode,
            device,
            number_of_samples,
            number_of_iterations)
    de.de(fobj=DE.bert_fobj,
          bounds=[(-1, 1)],
          model=model)

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
    print(f'Exact match: {metrics_["exact_match"]}, F1 score: {metrics_["f1"]}')

    print('')
    validation_time = format_time(time.time() - t0)

    print("  Validation took: {:}".format(validation_time))

    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_train_time_start)))
