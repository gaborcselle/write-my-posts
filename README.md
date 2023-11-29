# write-my-posts
Write Twitter / X / Mastodon / Threads style posts in my voice.

*Watch me code this up on [YouTube](https://www.youtube.com/watch?v=51DWARJckL4)! I'm intending to start this project at 9 am PT on Nov 29, 2023, and I have set myself a deadline of 12 noon PT: 3 hours total.*

# Learnings

You can try the end result at [write-my-posts.gaborcselle.com](https://write-my-posts.gaborcselle.com/). You can have GPT write posts in your voice by following the "How to Run" section below.

1. *Content:* 25% of the total time of the livestream was watching the OpenAI finetune console. That is *not* amazing content.
2. *Viewership:* We had a total of 16 viewers overall and no chats, confirming that this is not great content.
3. *Data procurement:* The approach of using my downloadable Twitter archive did work, that was surprisingly easy to reformat into training / test data. (Grr Twitter for not providing a free Twitter read API.)
4. *Finetune on small amounts of data:* The overall approach of finetuning GPT-3.5 to match my Twitter tone did work, even with a small sample of 200 tweets train / 40 tweets validation we got to a pretty good point in matching my tone, even if the content made no sense.
5. *Finetune on large amounts of data:* I then trained a model on all my tweets that weren't replies (~4000 train/~500 validation) and it's a little bit better but not phenomenally so. The posts it produces are a bit more cohesive than the smaller model.

# Objective
Write Twitter / X / Mastodon / Threads posts for me, in my voice.

# How
Download all my previous tweets, and finetune GPT-3.5-turbo on it. Then build a simple web app to generate posts. I'll use topics as the prompt. The topics will be extracted from my tweets using `tweetnlp`.

# Motivation
We did a crappy version of AI generated posts with our now-defunct social platform Pebble. We used few-shot prompting, and our users complained that the posts didn't sound like them at all. It let to an event we called "Ideageddon" - the largest user revolt we ever had. You can read more about the user revolt at [blog post](https://medium.com/gabor/from-t2-to-pebble-the-rise-challenges-and-lessons-of-building-a-twitter-alternative-553652f1d1e7).

I'm curious if a more sophisticated approach to post generation by finetuning GPT-3.5-turbo might result in better posts. That's why I'm building this.

I was also inspired by [this video by Pieter Levels](https://www.youtube.com/watch?v=6reLWfFNer0&t=657s) who advocates for livestreaming yourself coding as that makes you dramatically more productive. Thus the YouTube stram.

# How do we get the training data?
I looked a bit into this.
1. My original idea was to use the X/Twitter API to pull my tweets. But Free tier of the Twitter API will no longer let you pull tweets. You have to pay $100/month for that. *So that’s out.*
2. I looked into using my Pebble archive that I downloaded in the final days of Pebble. But most of you don't have that. *So that’s out.*
3. I found a workaround: [X/Twitter allows you to download your archive](https://help.twitter.com/en/managing-your-account/how-to-download-your-twitter-archive). They make this inconvenient by making you wait for it. Once you get it, it contains a beautiful archive of your tweets. I'll use that, and you'll be able to use it too. It contains about 10.9k tweets I've written.

# What LLM technique do we use?
At Pebble we used few-shot prompting … it didn't work well (see above).

[This OpenAI Dev Day video](https://www.youtube.com/watch?v=ahnGLM-RC1Y&t=1373s) explains when to use few-shot vs. RAG vs. finetune. (There’s an IMHO funny joke at 32:53.)

# What do we prompt on?
You need to give GPT somehting to write about. For each of my tweets, my plan is to use [tweetnlp](https://github.com/cardiffnlp/tweetnlp) to do NER (named entity recognition) and then use the entity tags as the prompt.

# What model do we use?
GPT-3.5-turbo, since OpenAI allows easy fine-tuning.

We'll base the finetuning code on [this notebook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_finetune_chat_models.ipynb) - note that this notebook has a bug in it, I sent a PR to OpenAI to fix it which is [here](https://github.com/openai/openai-cookbook/pull/885) and I'll be using my fork of the repo until that's merged.

# Rough plan of attack

1. Use tweetnlp to extract entities from my tweets
2. 80/20 split on the posts, 80% for training, 20% for validation
3. Finetune GPT-3.5-turbo on the training set
4. Plug the finetuned model into the Next.js app

# Deadline
I'm intending to start this project at 9 am PT on Nov 29, and I have set myself a deadline of 12 noon PT: 3 hours total.

# Layout
- `app/` - contains the Next.js app to interactively generate posts, deployed to Vercel at [https://write-my-posts.gaborcselle.com/](https://write-my-posts.gaborcselle.com/)
- `train/` - Python code to fine-tune GPT-3.5-turbo on my tweets

# How to Run
(This is a work in progress, I'll complete this after the livestream.)
If you want to try this yourself: 

1. Request your Twitter archive from [https://twitter.com/settings/your_twitter_data](https://twitter.com/settings/your_twitter_data)
2. Install `train/requirements.txt`
3. `cp train/config.example.py train/config.py` and fill in your OpenAI API key
4. `cp env.local.example env.local` and fill in your OpenAI API key
5. Copy your Twitter archive's `tweets.js` into `train/data/tweets.js`
6. Run `train/prep_finetune_data.py` to generate train and validation files for your finetune of GPT-3.5-turbo
7. Run `train/run_finetune.py` to finetune GPT-3.5-turbo
8. After finetune completed, plug your model ID into `app/api/chat/route.ts`
9. `npm install` or equivalent to install the Next.js app
10. `npm run dev` to run the Next.js app


# Source for Next.js app

The Next.js example was forked from an example app written by Vercel at [vercel/ai](https://github.com/vercel/ai/tree/main/examples/next-openai)
