import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

pat1 = r'@[A-Za-z0-9_]+' #mentions
pat2 = r'https?://[A-Za-z0-9./]+' # URLs
www_pat = r'www.[^ ]+'
combined_pat = r'|'.join((pat1, pat2))

def clean_tweet(tweet):
    soup = BeautifulSoup(tweet, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    stripped = re.sub(www_pat, '', stripped)
    only_ascii = re.sub(r'[^\x00-\x7F]+', ' ', stripped)
    return only_ascii

def process_training():
    cols = ['sentiment','id','date','query_string','user','text']
    df = pd.read_csv(
        "/Users/acko034/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv",
        header=None, names=cols, encoding = "ISO-8859-1"
    )
    df.drop(['id','date','query_string','user'],axis=1,inplace=True)

    rows = (df.shape)[0]
    print("Cleaning and parsing the tweets...")
    clean_tweet_texts = []

    start=datetime.now()
    for i in range(0, rows):
        if((i + 1) % 100000 == 0):
            print("Tweets %d of %d has been processed" % ( i+1, rows))
        clean_tweet_texts.append(clean_tweet(df['text'][i]))

    print("Time taken: ", datetime.now()-start)

    clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
    clean_df['sentiment'] = df.sentiment
    print(clean_df.info())
    clean_df.to_csv('clean_tweet.csv',encoding='utf-8')

# process_training()

df = pd.read_csv("clean_tweet.csv", header=[0], dtype={"text": np.str_, "sentiment": np.int64}, index_col=0)
neg_tweets = df[df['sentiment'] == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()





