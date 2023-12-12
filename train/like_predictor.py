import pandas as pd
import json
from datetime import datetime
from sklearn import datasets
from transformers import XLMRobertaTokenizer, XLMRobertaModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Check if embeddings.pkl file exists
if os.path.exists('like_predictor_data.pkl'):
    df = pd.read_pickle('like_predictor_data.pkl')
else:
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
        timestamp = created_at.timestamp()
        tweets_data.append({
            'full_text': tweet['tweet']['full_text'],
            'timestamp': timestamp,
            'lang': tweet['tweet']['lang'],
            'like_count': tweet['tweet']['favorite_count'],
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute            
        })

    df = pd.DataFrame(tweets_data)
    # Calculate and store the embeddings
    print('Calculating embeddings...')
    
    print(df.head())

    # apply embed_text function to the full_text column and show a progress bar
    tqdm.pandas()
    df['full_text_embedding'] = df['full_text'].progress_apply(embed_text)
    
    df.to_pickle('like_predictor_data.pkl')
    
# Define the model
class TweetLikePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = model
        self.regressor = nn.Linear(model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, timestamp, lang):
        outputs = self.text_embedding(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

# Instantiate the model
tweet_model = TweetLikePredictor()

train_dataset, validation_dataset = train_test_split(df, test_size=0.2, random_state=42)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
)

# Create trainer and train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
trainer.train()

# Save model to Hugging Face
model.save_pretrained("./models")
tokenizer.save_pretrained("./models")

model.push_to_hub('gaborcselle/tweet-likes-predictor')