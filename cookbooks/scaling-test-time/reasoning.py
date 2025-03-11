import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def confidence(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:

    """
    Calculate the confidence score (Δ) as specified in the paper.
    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer
    
    Returns:
        Confidence score (Δ)
    """

    confidence_sum = 0.0
    valid_tokens = 0

    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0 
        else:
            confidence_sum += 1.0 
        valid_tokens += 1
    
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0
