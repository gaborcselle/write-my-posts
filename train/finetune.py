import openai
import config
import json

openai.api_key = config.OPENAI_API_KEY

# open tweets.js and remove the prefix of "window.YTD.tweets.part0 = "
# then parse the contents
with open('data/tweets.js', 'r') as f:
    tweets = f.read()
    tweets = tweets.replace('window.YTD.tweets.part0 = ', '')

# parse the remaining string as JSON
tweets = json.loads(tweets)

# extract the full_text from all the tweets
# and put them into a list
training_data = [tweet['tweet']['full_text'] for tweet in tweets]

system_message = "You write X/Twitter posts about topics provided by the user."

def prepare_example_conversation(topic, result):
    messages = []
    messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": topic})
    messages.append({"role": "assistant", "content": result})
    return {"messages": messages}

import pprint
pprint.pprint(prepare_example_conversation("topic", training_data[0]))
