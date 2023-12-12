import pandas as pd
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import DataCollatorWithPadding
import torch.nn as nn
import evaluate

PKL_FILE_NAME = 'like_predictor_data.pkl'

id2label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5 : '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10+'}
label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,  '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10+': 10}

model = model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=11, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def get_data():
    # Check if embeddings.pkl file exists
    if os.path.exists(PKL_FILE_NAME):
        print('Loading embeddings from file...')
        df = pd.read_pickle(PKL_FILE_NAME)
    else:
        print('Loading tweets and embedding...')
        # Load your tweets.js file
        with open('data/tweets.js', 'r') as f:
            tweets = f.read()
            tweets = tweets.replace('window.YTD.tweets.part0 = ', '')

            # parse the remaining string as JSON
            data = json.loads(tweets)

        # Extract relevant fields and convert 'created_at' to timestamp
        tweets_data = []
        for tweet in data:
            created_at = datetime.strptime(tweet['tweet']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            year = created_at.year
            month = created_at.month
            day = created_at.day
            hour = created_at.hour
            minute = created_at.minute
            timestamp = created_at.timestamp(),
            # cap the label at 10, 10 means 10 or more likes
            label = 10 if int(tweet['tweet']['favorite_count']) > 10  else int(tweet['tweet']['favorite_count'])
            tweets_data.append({
                'full_text': tweet['tweet']['full_text'],
                'timestamp': timestamp,
                'lang': tweet['tweet']['lang'],
                'like_count': tweet['tweet']['favorite_count'],
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'labels': label
            })

        df = pd.DataFrame(tweets_data)
        # Calculate and store the embeddings
        print('Saving...', PKL_FILE_NAME)

        df.to_pickle(PKL_FILE_NAME)
    
    return df

if __name__ == '__main__':
    data = get_data()

    # texts
    texts = data['full_text'].tolist()
    encoded_inputs = tokenizer(texts)

    # labels
    labels = data['labels'].tolist()
    
    ds = Dataset.from_dict({'input_ids': encoded_inputs['input_ids'], 'attention_mask': encoded_inputs['attention_mask'], 'labels': labels})   

    print("ds[0]", ds[0])

    # Split the dataset into a training and a validation dataset
    train_dataset = ds.train_test_split(test_size=0.2, seed=42)['train']
    val_dataset = ds.train_test_split(test_size=0.2, seed=42)['test']
    print("train_dataset[0]", train_dataset[0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./tweet_like_predictor_od',          # output directory
        hub_model_id = 'gaborcselle/tweet-like-count-predictor',             # model name
        logging_dir='./logs',            # directory for storing logs
        push_to_hub=True
    )

    # Define the trainer parameters
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model('./tweet_like_predictor')

    # upload the model to Hugging Face
    trainer.push_to_hub('gaborcselle/tweet_like_predictor')





