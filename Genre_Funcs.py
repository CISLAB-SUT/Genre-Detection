from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from pyonion.remover import ListCorpusProvider, DuplicateRemover, CleaningMode
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os, pickle, random,time, math, re, json, inspect, ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text as TXT
from collections import defaultdict, Counter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch 

def compute_metrics(eval_pred):
    '''
    Calculates metrics for eval during training

    Parameters:
        eval_pred(tuple): Contains predicted logits and the  label
    
    Returns:
        dict: A dictionary containing eval results
    '''
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  
    preds = (probs > 0.5).astype(int)
    exact_match = (preds == labels).all(axis=1).mean()
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    h_loss = hamming_loss(labels, preds)
    
    return {
        "exact_match": exact_match,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "hamming_loss": h_loss
    }

def dynamic_text(tokenizer, domain, title, size, text, token_limit=512):
    '''
    Creates model inputs and chunks them based on tokenizers token limit

    Parameters:
        tokenizer(hf.tokenizer): Tokenizer of the model
        domain(str): Category of the input page
        title(str): Title of the input page
        size(str): Size of the input page
        text(str): Full text content of the input page
        token_limit(int): Maximum number of tokens per input chunk
    
    Returns:
        chunks(list): A list of strings each containing domain, title, size and chunk of text
    '''
    # Tokenize the fixed prefix
    prefix = f"{domain} [SEP] {title} [SEP] {size} [SEP]"
    prefix_ids = tokenizer(prefix, truncation=True, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_ids)

    # Remaining tokens for text
    max_text_len = token_limit - prefix_len
    text_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    # Split into chunks
    chunks = []
    for i in range(0, len(text_ids), max_text_len):
        chunk_ids = text_ids[i:i + max_text_len]
        input_ids = prefix_ids + chunk_ids
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        chunks.append(input_text)
    return chunks

def deduplication(text, n_gram=5, threshold=0.5):
    '''
    Finds and removes paragraphs that have similar ngrams

    Parameter:
        text(str): Full text content of the input page
        n_gram(int): The number of ngrams that are used as the metric for similarity
        threshold(float): Percentage of how similar paragraphs need to be in order to be considered duplicate
    
    Returns:
        clean_corpus(list): List of paragraphs without duplicates
    '''
    parags = text.split("\n")
    corpus = ListCorpusProvider(parags)
    remover = DuplicateRemover(n_gram=n_gram)
    duplicated_ngrams = remover.find_duplicated_ngrams(corpus)

    clean_text = remover.iter_clean_text(
        corpus, duplicated_ngrams,
        threshold=threshold,
        mode=CleaningMode.FIRST
    )

    clean_corpus = [txt for txt, sim in clean_text if len(txt)>0]
    return clean_corpus

def preprocess(urls, page_data, tokenizer, labels_converted, token_limit=512, one_hot=True):
    '''
    Preprocess input page into proper inputs for the model.

    Parameters:
        urls(pd.DataFrame): A datframe containing html ids and labels
        page_data(pd.DataFrame): A dataframe containg html ids, titles and texts
        tokenizer(hf.tokenizer): Tokenizer of the model
        labels_converted(dict): A dictionary mapping string labels to integer labels
        token_limit(int): Maximum number of tokens per input chunk. used for dynamic_text func
        one_hot(bool): Converts integer labels to one-hot if True

    Returns:
        full_dataset(list): A list that contains page chunks inside a dictionary of page id, chucked text and page label.
    '''
    urls_pp = urls.copy()

    page_data['length'] = page_data['text'].apply(len)
    page_data['size'] = pd.qcut(page_data['length'], q=3, labels=['short', 'medium', 'long'])

    urls_pp['type'] = urls_pp['type'].apply(
            lambda x: [labels_converted[label.strip()] for label in str(x).split(",")]
        )

    if one_hot:
        for index, row in urls_pp.iterrows():
            new_label = [0]*9
            for l in row['type']:
                new_label[l] = 1
            urls_pp.at[index, 'type'] = new_label

    
    
    full_dataset = []
    for index, row in tqdm(urls_pp.iterrows(), total=len(urls_pp), desc="Preparing datasets"): 
        try:
            traf_row = page_data[page_data['html_id'] == row['id']]
            domain = row['category']

            title = traf_row['title'].iloc[0]
            size = traf_row['size'].iloc[0]
            text = traf_row['text'].iloc[0]
            text_dedup = " ".join(deduplication(text))
            true_label = row['type']

            batch_text = dynamic_text(tokenizer, domain, title, size, text_dedup, token_limit=token_limit)
            for td in batch_text:
                full_dataset.append({
                        "id": row['id'],
                        "text": td,
                        "label": true_label
                    })
        except Exception as e:
            print(f"Error processing row {index} with id {row['id']}: {e}")
    return full_dataset

class Dataset_Config(Dataset):
    '''
    Prepares text classification data using into pytorch's Dataset format.

    Parameters:
        data (list[dict]): A list of dictionaries where each item contains 'text' and 'label'.
        tokenizer (transformers.Tokenizer): The tokenizer used for encoding the text.
        max_len (int): The maximum token length for truncation (default is 512).

    Returns: 
        dict: A dataset compatible with pytorch.
    '''
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encodings = self.tokenizer(text, truncation=True, padding='max_length', return_tensors="pt").to("cuda")

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float),
        }

def train_model(model, tokenizer, train_dataset, test_dataset, fine_type):
    '''
    Trains model on input dataset

    Parameters:
        model(hf.model): The main model for finetuning
        tokenizer(hf.tokenizer): Tokenizer of the model
        train_dataset(torch.dataset): Training dataset, converted to torch for arg matching
        test_dataset(torch.dataset): Evaluation dataset, converted to torch for arg matching
        fine_type(str): Unique name for the folder of created finetuned checkpoint
    '''
    # Setting up Fine-tune training configurations
    training_args = TrainingArguments(
        output_dir=f"./ensemble_models/{fine_type}_genre_chkp",
        fp16=True,
        eval_strategy="epoch",
        # eval_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.08,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        learning_rate=4e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        weight_decay=0.01,
        logging_dir=f"./ensemble_models/{fine_type}_genre_chkp/logs",
        logging_strategy="epoch",

    )
    
    # Set up Trainer for fine-tuneing
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"Model performance after fine-tuning: {metrics}")

    # Extract the log history
    logs = trainer.state.log_history
    # Group losses and F1 scores by epoch
    train_losses = defaultdict(list)
    eval_losses = {}
    eval_f1s = {}

    for log in logs:
        epoch = log.get("epoch")
        if epoch is not None:
            if "loss" in log:
                train_losses[epoch].append(log["loss"])
            if "eval_loss" in log:
                eval_losses[epoch] = log["eval_loss"]
            if "eval_f1" in log:
                eval_f1s[epoch] = log["eval_f1"]

    # Compute average train loss per epoch
    epochs = sorted(set(train_losses.keys()) | set(eval_losses.keys()) | set(eval_f1s.keys()))
    avg_train = [np.mean(train_losses[ep]) if ep in train_losses else None for ep in epochs]
    avg_eval = [eval_losses.get(ep, None) for ep in epochs]
    avg_f1 = [eval_f1s.get(ep, None) for ep in epochs]

    # Plot Loss graph
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, avg_train, label="Train Loss", marker='o', color='tab:blue')
    plt.plot(epochs, avg_eval, label="Eval Loss", marker='x', color='tab:orange')
    plt.plot(epochs, avg_f1, label="Eval F1", marker='d', color='tab:green')

    plt.xlabel("Checkpoints")
    plt.ylabel("Evaluation")
    plt.title(f"Loss and Eval F1 per Epoch {fine_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss_f1_graph {fine_type}.png")


def test_model(model, tokenizer, eval_dataset, eval_type="prod", ensemble=False):
    '''
    Evalutes the fine-tuned model. Calculates the mean of the logits of each chunk for a page. Resulting logits is the final predition for the input page.
    If eval_type='one_hot', number of predicted labels are same as number of labels in input page.
    But if eval_type!='one_hot' then numeber of predicted labels are always 2.

    Parameters:
        model(hf.model): The main model for finetuning
        tokenizer(hf.tokenizer): Tokenizer of the model
        eval_dataset(list): A list that contains test page chunks inside a dictionary of page id, chucked text and page label. 
        eval_type(str): Whether to use one-hot evaluation metric (for paper) or more forgiving evaluation metric.
                        Note that of 'eval_type'='one_hot' it must be preprocessing(one_hot=True)!
        ensemble(bool): Whether to send the labels and logits to ensemble function or directly return predicted labels
    '''
    logits_sum = torch.zeros(9, device="cuda")
    total_response = []
    total_labels = []
    total_labels_ensemble = []
    top2_list = []
    total_logits = []

    eval_dataset = sorted(eval_dataset, key=lambda x: x['id'])
    assert eval_dataset == sorted(eval_dataset, key=lambda x: x['id'])

    current_id = eval_dataset[0]['id']
    id_count = 0
    current_labels = None
    for data in tqdm(eval_dataset, total=len(eval_dataset), desc="Evaluating test data"):
        start_time = time.time()
        if data['id'] == current_id:
            inputs = tokenizer(data['text'], return_tensors="pt", truncation=True, padding=True).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)
            logits_sum += logits
            id_count += 1
            current_labels = data['label']
        else:
            true_label = current_labels

            k_num = sum(true_label)

            # Compute mean logits for the previous ID
            logits_mean = logits_sum / id_count
            total_logits.append(logits_mean.detach().cpu())

            top2 = torch.topk(logits_mean, k=k_num).indices.tolist()

            # Decide predicted class
            predicted_class = [0]*9
            predicted_class = [1 if i in top2 else 0 for i in range(len(predicted_class))]
            

            if eval_type != "one_hot":
                predicted_class = top2[0]
                true_label=current_labels[0]
                for L in data['label']:
                    if L in top2:
                        predicted_class = L
                        true_label = L
                        break
            
            total_response.append(predicted_class)
            total_labels.append(true_label) 
            total_labels_ensemble.append(current_labels)

            # Reset for next ID
            current_id = data['id']
            logits_sum = torch.zeros(9, device="cuda")
            id_count = 0

            # process the new ID
            inputs = tokenizer(data['text'], return_tensors="pt", truncation=True, padding=True).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            logits_sum += outputs.logits.squeeze(0)
            id_count += 1
            current_labels = data['label']

    # Handle last ID
    true_label = current_labels
    k_num = sum(true_label)
    logits_mean = logits_sum / id_count
    total_logits.append(logits_mean.detach().cpu())

    
    top2 = torch.topk(logits_mean, k=k_num).indices.tolist()

    predicted_class = [0]*9
    predicted_class = [1 if i in top2 else 0 for i in range(len(predicted_class))]

    if eval_type != "one_hot":
        predicted_class = top2[0]
        true_label=current_labels[0]
        for L in data['label']:
            if L in top2:
                predicted_class = L
                true_label = L
                break
            
    total_response.append(predicted_class)
    total_labels.append(true_label)
    total_labels_ensemble.append(current_labels)

    if ensemble:
        return total_labels_ensemble, total_logits

    return total_response, total_labels



def ensemble_predictions(checkpoint_paths, models_names, urls, page_data, labels_converted, test_ids, one_hot=False, max_len=512):
    '''
    Evaluates multiple chekpoints and returns predicted labels based on averaged logits

    Parameters:
        checkpoint_paths(list): A list of paths to checkpoints
        models_names(list): A list of name of the chosen checkpoint
        urls(pd.DataFrame): A datframe containing html ids and labels
        page_data(pd.DataFrame): A dataframe containg html ids, titles and texts
        labels_converted(dict): A dictionary mapping string labels to integer labels
        test_ids(list): List of test html ids        
        one_hot(bool): Converts integer labels to one-hot if True
        max_len (int): The maximum token length for truncation (default is 512).
    
    Returns:
        totla_reponses(list): List of predicted labels

    '''
    all_logits = []
    for index, model_checkpoint in enumerate(checkpoint_paths):
        print(f"\nEvaluating model: {models_names[index]}\n")
        torch.cuda.empty_cache()
        model_name = models_names[index]
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                problem_type="multi_label_classification", num_labels=9).to("cuda") 
        
        full_dataset_TEST = preprocess(urls, page_data, tokenizer, labels_converted, token_limit=max_len, one_hot=one_hot)
        train_dataset, test_dataset = [],[]

        for pages in full_dataset_TEST:
            if pages['id'] not in test_ids:
                train_dataset.append(pages)
            else:
                test_dataset.append(pages)

        if one_hot:
            eval_type = 'one_hot'
        else: eval_type = 'prod'
        
        model.eval()

        total_labels, total_logits = test_model(model, tokenizer, eval_dataset=test_dataset, eval_type=eval_type, ensemble=True)
        total_logits = torch.stack(total_logits)
        all_logits.append(total_logits)

    all_logits = torch.stack(all_logits)
    print(f"\nAll logits shape: {all_logits.shape}")
        
    total_response = [] 
    total_labels_new = []
    for i in range(all_logits.shape[1]):
        logits_mean = torch.mean(all_logits[:, i, :], dim=0)

        k_num = sum(total_labels[i])
        topk_indices = torch.topk(logits_mean, k=k_num).indices.tolist()

        response = [0] * 9
        for idx in topk_indices:
            response[idx] = 1     
            
        total_response.append(response)
        total_labels_new.append(total_labels[i])

    return total_response, total_labels_new
