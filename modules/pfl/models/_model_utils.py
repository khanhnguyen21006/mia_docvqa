import random, re
import torch
import numpy as np


def get_extractive_confidence(outputs):
    bs = len(outputs['start_logits'])
    start_idxs = torch.argmax(outputs.start_logits, axis=1)
    end_idxs = torch.argmax(outputs.end_logits, axis=1)

    answ_confidence = []
    for batch_idx in range(bs):
        conf_mat = np.matmul(np.expand_dims(outputs.start_logits.softmax(dim=1)[batch_idx].unsqueeze(dim=0).detach().cpu(), -1),
                             np.expand_dims(outputs.end_logits.softmax(dim=1)[batch_idx].unsqueeze(dim=0).detach().cpu(), 1)).squeeze(axis=0)

        answ_confidence.append(
            conf_mat[start_idxs[batch_idx], end_idxs[batch_idx]].item()
        )

    return answ_confidence


def get_generative_confidence(output):
    batch_logits = torch.stack(output.scores, dim=1)[:, :-1, :]  # b x s x V and dropping EOS token
    decoder_output_confs = torch.amax(batch_logits.softmax(dim=-1), dim=2)
    confidences = decoder_output_confs.prod(dim=1)  # b
    return confidences.tolist()
