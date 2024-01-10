import evaluate
import datetime
import numpy as np
import collections


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def predict_answers_and_evaluate(start_logits, end_logits, eval_set, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["base_id"]].append(idx)

    n_best = 20
    max_answer_length = 30
    predicted_answers = []

    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logit).tolist()[::-1][:n_best]
            end_indexes = np.argsort(end_logit).tolist()[::-1][:n_best]

            for start_index in start_indexes:
                for end_index in end_indexes:

                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answers.append({
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    metric = evaluate.load("squad")

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]

    metric_ = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    return predicted_answers, metric_


class Utils:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def train_data_preprocess(self, examples):

        def find_context_start_end_index(sequence_ids):
            token_idx = 0
            while sequence_ids[token_idx] != 1:
                token_idx += 1
            context_start_idx = token_idx

            while sequence_ids[token_idx] == 1:
                token_idx += 1
            context_end_idx = token_idx - 1
            return context_start_idx, context_end_idx

        questions = [q.strip() for q in examples["question"]]
        context = examples["context"]
        answers = examples["answers"]

        inputs = self.tokenizer(
            questions,
            context,
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        start_positions = []
        end_positions = []

        for i, mapping_idx_pairs in enumerate(inputs['offset_mapping']):
            context_idx = inputs['overflow_to_sample_mapping'][i]

            answer = answers[context_idx]
            answer_start_char_idx = answer['answer_start'][0]
            answer_end_char_idx = answer_start_char_idx + len(answer['text'][0])

            tokens = inputs['input_ids'][i]
            sequence_ids = inputs.sequence_ids(i)

            context_start_idx, context_end_idx = find_context_start_end_index(sequence_ids)

            context_start_char_index = mapping_idx_pairs[context_start_idx][0]
            context_end_char_index = mapping_idx_pairs[context_end_idx][1]

            if (context_start_char_index > answer_start_char_idx) or (
                    context_end_char_index < answer_end_char_idx):
                start_positions.append(0)
                end_positions.append(0)

            else:

                idx = context_start_idx
                while idx <= context_end_idx and mapping_idx_pairs[idx][0] <= answer_start_char_idx:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end_idx
                while idx >= context_start_idx and mapping_idx_pairs[idx][1] > answer_end_char_idx:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")

        base_ids = []

        for i in range(len(inputs["input_ids"])):
            base_context_idx = sample_map[i]
            base_ids.append(examples["id"][base_context_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["base_id"] = base_ids
        return inputs
