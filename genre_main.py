from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from Genre_Funcs import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch 


# Read url and labels dataset
urls = pd.read_excel("genre_final.xlsx").head(2000)

# # Read page contents dataset
page_data = pd.read_excel("page_data.xlsx")


# Test IDs
test_ids = np.load("genre_test_ids_new.npy")
# test_ids = np.load('300_pages.npy')
# np.random.shuffle(test_ids_new)
# test_ids = np.append(test_ids, test_ids_new[:100])
print(len(test_ids))


labels_converted = {
    "0": 0,
    "research": 1,
    "news": 2,
    "report": 3,
    "project": 4,
    "product": 5,
    "people": 6,
    "event": 7,
    "grant": 8
}


train = False
test = True
ensemble = True

fine_type = 'distil'
epoch_number = -1
max_len = 512
full_dataset = None
########## GENRE DETECTION FINE-TUNING ##########
if train:
    torch.cuda.empty_cache()
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
            problem_type="multi_label_classification", num_labels=9).to("cuda")
    tokenizer.model_max_length = max_len

    for param in model.distilbert.parameters():
        param.requires_grad = False
    for param in model.distilbert.transformer.layer[-2:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    if not full_dataset:
        full_dataset = preprocess(urls, page_data, tokenizer, labels_converted, token_limit=max_len, one_hot=True)
        train_dataset, test_dataset = [],[]

        for pages in full_dataset:
            if pages['id'] not in test_ids:
                train_dataset.append(pages)
            else:
                test_dataset.append(pages)
        print("\nTrain Dataset Shape:", len(train_dataset))
        print("Test Dataset Shape:", len(test_dataset), "\n")

    train_dataset_loaded = Dataset_Config(train_dataset, tokenizer, max_len=max_len)
    test_dataset_loaded = Dataset_Config(test_dataset, tokenizer, max_len=max_len)

    ##  TRAINING THE MODEL ##
    train_model(model, tokenizer, train_dataset_loaded, test_dataset_loaded, fine_type)
    print("Train finished!\n\n\n")



####^^#### GENRE DETECTION EVALUATION ####^^####
if test:
    if ensemble:
        model_paths = "./"
        models_names = os.listdir(model_paths)
        checkpoint_paths = [
            os.path.join(model_paths, model_folder, checkpoint)
            for model_folder in os.listdir(model_paths)
            if os.path.isdir(os.path.join(model_paths, model_folder))
            for checkpoint in os.listdir(os.path.join(model_paths, model_folder))
            if checkpoint.startswith("checkpoint-")
        ]
        print("Checkpoints found for ensemble::\n", checkpoint_paths)

    else:
        output_dir = f"./{fine_type}_genre_chkp/"
        checkpoints = sorted([int(d.replace("checkpoint-", "")) for d in os.listdir(output_dir) if d.startswith("checkpoint-")])

        print("\nLoaded Checkpint:", f"checkpoint-{checkpoints[epoch_number]}\n")

        torch.cuda.empty_cache()
        model_name = f"{output_dir}checkpoint-{checkpoints[epoch_number]}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                problem_type="multi_label_classification", num_labels=9).to("cuda")
        model.eval()


    if not ensemble:
        if not full_dataset:
            full_dataset = preprocess(urls, page_data, tokenizer, labels_converted, token_limit=max_len, one_hot=True)
            train_dataset, test_dataset = [],[]

            for pages in full_dataset:
                if pages['id'] not in test_ids:
                    train_dataset.append(pages)
                else:
                    test_dataset.append(pages)

        torch.cuda.synchronize()
        total_response, total_labels = test_model(model, tokenizer, eval_dataset=test_dataset, eval_type='one_hot')
        

    if ensemble:
        total_response, total_labels = ensemble_predictions(checkpoint_paths, models_names,
                                        urls, page_data, labels_converted, test_ids, one_hot=True, max_len=512)

    if len(total_labels) == len(total_response):
        print(classification_report(total_labels, total_response, digits=2))
    else: 
        raise ValueError(f"labels and response don't match {len(total_labels)} != {len(total_response)}")
    
    prec = precision_score(total_labels, total_response, average="macro", zero_division=0)
    rec = recall_score(total_labels, total_response, average="macro", zero_division=0)
    f1 = f1_score(total_labels, total_response, average="macro", zero_division=0)
    acc = accuracy_score(total_labels, total_response)
    print("  > Precision score: ", prec)
    print("  > Recall Score: ", rec)
    print("  > F1-Score: ", f1)
    print("  > Accuracy score: ", acc)