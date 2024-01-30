import torch
import time
import torch.nn as nn
import numpy as np
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from src.utils import Utils, format_time, predict_answers_and_evaluate


class CutterDistilBERT:
    def __init__(self, model):
        self.model = model
        self.custom_qa_output = None

    def set_linear_layer(self, per_model):
        self.custom_qa_output = per_model

    def __call__(self,
                 input_ids,
                 attention_mask,
                 start_positions,
                 end_positions,
                 return_dict,
                 for_embs):

        distilbert_output = self.model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.model.dropout(hidden_states)  # (bs, max_query_len, dim)
        if self.custom_qa_output:
            logits = self.custom_qa_output(hidden_states)
        else:
            logits = self.model.qa_outputs(hidden_states)  # (bs, max_query_len, 2)

        if for_embs:
            return logits
        else:
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
            end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=distilbert_output.hidden_states,
                attentions=distilbert_output.attentions,
            )

    def evalute(self,
                eval_dataloader,
                validation_processed_dataset,
                dataset,
                device):
        self.model.eval()

        start_logits, end_logits = [], []
        for step, batch in enumerate(eval_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                result = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask, return_dict=True)

            start_logits.append(result.start_logits.cpu().numpy())
            end_logits.append(result.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)

        answers, metrics_ = predict_answers_and_evaluate(start_logits, end_logits,
                                                         validation_processed_dataset,
                                                         dataset["validation"])
        print(f'### FT Exact match: {metrics_["exact_match"]}, F1 score: {metrics_["f1"]}')
