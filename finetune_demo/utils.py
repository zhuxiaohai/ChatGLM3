import torch
from torch.nn import CrossEntropyLoss
import numpy as np


def mask_prompt(sequences, context_lengths, is_right_padded=True):
    sequences = sequences.clone()
    # Step 1: Calculate effective context lengths
    # Find indices of first non-zero element in each row (accounting for left padding)
    if is_right_padded:
        effective_context_lengths = context_lengths
    else:
        padding_lengths = sequences.argmax(dim=1)
        effective_context_lengths = context_lengths + padding_lengths

    # Step 2: Create a range tensor
    range_tensor = torch.arange(sequences.size(1)).expand_as(sequences).to(sequences.device)

    # Step 3: Expand effective context lengths for broadcasting
    expanded_effective_context_lengths = effective_context_lengths.unsqueeze(1).expand_as(sequences)

    # Step 4: Generate mask where range is less than expanded effective context lengths
    mask = range_tensor < expanded_effective_context_lengths

    # Apply mask to set specified positions in sequences to 0
    sequences[mask] = 0

    return sequences

def compute_ppl(
        predictions, model, tokenizer,
        conditional=True, collator=None, prompt_length=None, is_right_padded=False, add_special_tokens=True,
        batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
):
    model = model.to(device)

    if collator is None:
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
        loss_fct = CrossEntropyLoss(reduction="none")
    else:
        encodings = predictions
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)

    ppls = []
    data_tol = len(predictions)
    for start_index in range(0, data_tol, batch_size):
        end_index = min(start_index + batch_size, data_tol)
        if collator is None:
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )
            labels = encoded_batch
        else:
            data_batch = collator([encodings[i] for i in range(start_index, end_index)])
            encoded_batch = data_batch["input_ids"].to(device)
            attn_mask = data_batch["attention_mask"].to(device)
            labels = data_batch["labels"].to(device)
            if not conditional:
                attn_mask = torch.where(labels == -100, 0, 1).to(attn_mask.dtype).to(attn_mask.device)

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if collator is not None:
            loss_mask = torch.where(labels == -100, 0, 1).to(attn_mask.dtype).to(attn_mask.device)
        if prompt_length is not None:
            loss_mask = mask_prompt(attn_mask, prompt_length.to(device), is_right_padded)
        shift_loss_mask = loss_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_loss_mask).sum(1)
            / shift_loss_mask.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}