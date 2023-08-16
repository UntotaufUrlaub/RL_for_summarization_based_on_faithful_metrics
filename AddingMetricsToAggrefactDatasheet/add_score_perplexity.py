from add_score import add_score
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import warnings
warnings.filterwarnings('ignore')

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to('cuda')


def perplexity_batched(summ, doc, metric_type, also_return_negative):
    # uses a sliding window

    summary_tokens = tokenizer.tokenize(summ)
    # print("len summary", len(summary_tokens))
    text_prefix = doc + " "     # text_prefix = doc + " In summary: "
    prefix_length = len(tokenizer.tokenize(text_prefix))
    # print("len prefix", prefix_length)
    inputs = [None] * len(summary_tokens)
    token_indexes = [None] * len(summary_tokens)
    desired_tokens = [None] * len(summary_tokens)
    for i in range(len(summary_tokens)):
        if metric_type == "left_hand":
            summ_part = summary_tokens[:(i+1)]
        else:
            summ_part = summary_tokens
        desired_tokens[i] = summ_part[i]
        summ_part[i] = tokenizer.mask_token
        summ_part = tokenizer.convert_tokens_to_string(summ_part)
        text = text_prefix + summ_part

        tokens = tokenizer.tokenize(text)
        if len(tokens) > tokenizer.max_model_input_sizes[model_name]:
            start = max((prefix_length + i + 1)-tokenizer.max_model_input_sizes[model_name], 0)
            end = (prefix_length + i) + 1
            tokens = tokens[start:end]
            # print(f"truncated. start {start}, end {end}.")
        # print("text atm: ", tokenizer.convert_tokens_to_string(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        mask_token_index = tokens.index(tokenizer.mask_token)

        inputs[i] = input_ids
        token_indexes[i] = mask_token_index

    # # Convert the input IDs to a PyTorch tensor
    # inputs = [torch.tensor(input_array) for input_array in inputs]
    # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    # attention_mask = (inputs != 0).int()

    # Get the model's predictions for the input tensor
    batch_size = 2
    predictions = []
    for i in range(0, len(inputs), batch_size):
        upper_bound = min(i + batch_size, len(inputs))
        batch = inputs[i:upper_bound]
        batch = [torch.tensor(el) for el in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        print("batch shape:", batch.size())
        attention_batch = (batch != 0).int()

        if torch.cuda.is_available():
            batch = batch.to('cuda')
            attention_batch = attention_batch.to('cuda')

        with torch.no_grad():
            outputs = model(batch, attention_mask=attention_batch)
            del batch  # free gpu memory
            del attention_batch  # free gpu memory
            logits = outputs.logits
            predictions.extend([logits[i, token_indexes[i]] for i in range(len(logits))])
            torch.cuda.empty_cache()

    # print("predictions:", predictions)

    log_softmax = torch.nn.LogSoftmax(dim=0)
    log_probabilities = [log_softmax(t) for t in predictions]

    desired_token_indexs = tokenizer.convert_tokens_to_ids(desired_tokens)

    desired_token_log_probs = [log_probabilities[i][desired_token_indexs[i]].item() for i in range(len(desired_token_indexs))]

    log_prob = np.mean(desired_token_log_probs)

    res = torch.exp(torch.tensor(-log_prob)).item()

    if also_return_negative:
        return str({
            "normal": res,
            "negated": str(float(res)*-1),
        })
    else:
        return res


def perplexity(summ, doc, metric_type, also_return_negative):
    # uses a sliding window
    log_prob = None

    summary_tokens = tokenizer.tokenize(summ)
    # print("len summary", len(summary_tokens))
    text_prefix = doc + " "     # text_prefix = doc + " In summary: "
    prefix_length = len(tokenizer.tokenize(text_prefix))
    # print("len prefix", prefix_length)
    for i in range(len(summary_tokens)):
        if metric_type == "left_hand":
            summ_part = summary_tokens[:(i+1)]
        else:
            summ_part = summary_tokens
        desired_token = summ_part[i]
        summ_part[i] = tokenizer.mask_token
        summ_part = tokenizer.convert_tokens_to_string(summ_part)
        text = text_prefix + summ_part

        tokens = tokenizer.tokenize(text)
        if len(tokens) > tokenizer.max_model_input_sizes[model_name]:
            start = max((prefix_length + i + 1)-tokenizer.max_model_input_sizes[model_name], 0)
            end = (prefix_length + i) + 1
            tokens = tokens[start:end]
            # print(f"truncated. start {start}, end {end}.")
        # print("text atm: ", tokenizer.convert_tokens_to_string(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        mask_token_index = tokens.index(tokenizer.mask_token)

        # Convert the input IDs to a PyTorch tensor
        input_tensor = torch.tensor([input_ids])

        # Get the model's predictions for the input tensor
        with torch.no_grad():
            if torch.cuda.is_available():
                input_tensor = input_tensor.to('cuda')
            outputs = model(input_tensor)
            input_tensor.to('cpu')  # free gpu memory
            predictions = outputs.logits[0, mask_token_index]

        torch.cuda.empty_cache()

        # Calculate the log probabilities for each token
        log_softmax = torch.nn.LogSoftmax(dim=0)
        log_probabilities = log_softmax(predictions)

        # Find the index of the desired token in the vocabulary
        desired_token_index = tokenizer.convert_tokens_to_ids([desired_token])[0]

        # Calculate the log probability of the desired token
        desired_token_log_prob = log_probabilities[desired_token_index].item()
        # print(f"Log Probability of '{desired_token}': {desired_token_log_prob}, probability: "
        #       f"{torch.exp(torch.tensor(desired_token_log_prob)).item():.20f}")
        if log_prob is None:
            log_prob = desired_token_log_prob
        else:
            log_prob = log_prob + desired_token_log_prob

    log_prob = log_prob / len(summary_tokens)
    res = torch.exp(torch.tensor(-log_prob)).item()

    if also_return_negative:
        return str({
            "normal": res,
            "negated": str(float(res)*-1),
        })
    else:
        return res


for amount, seed in [
    # (1, 0),
    # (40, 0),
    (None, None),
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
            "psudo_perplexity_bert_lefthandContext": lambda summ, doc: perplexity(summ, doc, "left_hand", True),
            "psudo_perplexity_bert_fullContext": lambda summ, doc: perplexity(summ, doc, "right_hand", True),          # lower priority
            "psudo_perplexity_bert_lefthandContext_summaryOnly": lambda summ, doc: perplexity(summ, "", "left_hand", True),     # lower priority
            "psudo_perplexity_bert_fullContext_summaryOnly": lambda summ, doc: perplexity(summ, "", "right_hand", True),   # lower priority
#            "psudo_perplexity_bert_lefthandContext_swapped": lambda summ, doc: perplexity(doc, summ, "left_hand", True),
#            "psudo_perplexity_bert_fullContext_swapped": lambda summ, doc: perplexity(doc, summ, "right_hand", True),   # lower priority
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )
