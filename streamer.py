import streamlit as st
st.set_page_config(layout="wide")
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')
import plotly.graph_objects as go
import plotly.express as px
plt.style.use('seaborn')

#To Hide Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    cola, colb, colc = st.beta_columns([8,15,10])
    # colb, colc = st.beta_columns([15,10])
    with cola:
        st.title("Brand Management & Social Listening")
    with colc:
        from PIL import Image
        image = Image.open('Header.jpeg')
        st.image(image,use_column_width=True)


    # html_temp = """
	# <div style="background-color:tomato; text-align:center;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	# """
    # st.markdown(html_temp, unsafe_allow_html=True)
    # st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    ################# Twitter API Connection #######################
    consumer_key = "bUzuiuXLR4ZegX4M0Oiwk9NWJ"
    consumer_secret = "DE5QbQdJxC5tSU19uCxozmYAigO2YcrFnIGlGh3dTLZvQZoxwV"
    access_token = "883742475227078657-LLFboxHcfgw74A0EtBVXsZvfmcRZw18"
    access_token_secret = "NDUFBiyGU7Va6hD4ynIGI91R0ggMqI4qef9SuHPygNNrv"

    # Use the above credentials to authenticate the API.

    # auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth = tweepy.AppAuthHandler( consumer_key , consumer_secret )
    # auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    ################################################################

    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])


    # Write a Function to extract tweets:
    def get_tweets(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search, q=Topic,count=400, lang="en",exclude='retweets').items():
            #time.sleep(0.1)
            #my_bar.progress(i)
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            #df.to_csv("TweetDataset.csv",index=False)
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel

            i=i+1
            if i>Count:
                break
            else:
                pass
    # Function to Clean the Tweet.
    def clean_tweet(tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())


    # Funciton to analyze Sentiment
    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    #Function to Pre-process data for Worlcloud
    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
        ###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new



    with colb:
        # Collect Input from user :
        st.subheader("Enter a Brand/Product/Service which you want to check for")
        Topic = str()
        Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))
        # .........NEW...........
        # Count = str()
        Count = int(st.number_input("Enter Number"))

    if len(Topic) > 0 :

        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            # get_tweets(Topic , Count=1000)
            get_tweets(Topic, Count)



        # Call function to get Clean tweets
        df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))

        # Call function to get the Sentiments
        df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
        df["Polarity"] = df["Tweet"].apply(lambda x : TextBlob(x).sentiment.polarity)

        col1, col2 = st.beta_columns([2,1])

        with col1:
            st.success('Tweets have been Extracted !!!!')

            # See the Extracted Data :
            # if st.button("See the Extracted Data"):
                #st.markdown(html_temp, unsafe_allow_html=True)
            st.header("Below is the Extracted Data :")
            st.write(df.head(50))

            # Most Liked
            st.header("Most Popular Opinion")
            df1 = df[['User','Likes','Tweet']]
            # df1.set_index('Likes', inplace=True)
            # df1 = df1.sort_index(ascending=False)
            df1.sort_values(by=['Likes'], inplace=True, ascending=False)
            fig1 = go.Figure(data = [go.Table(
                columnorder = [1,2,3], columnwidth = [50,50,300],
                header = dict(values=list(df1.columns), line_color='darkslategray',
                fill_color='lightskyblue', align='center', font=dict(color='black', size=12),height=40),
                cells = dict(values = [df1.User, df1.Likes, df1.Tweet], line_color='darkslategray',
               fill_color='lightcyan', font_color='black', font_size=12, height=30)
            )])
            st.write(fig1)
            # st.write(df1.head(10))

            # TIME-Series
            st.header("Time-Series Analysis of Likes & Retweets")
            time_liked = pd.Series(data = df['Likes'].values, index=df['Date'])
            time_liked.plot(figsize=(20,7), label='Likes', legend=True)
            time_RT = pd.Series(data = df['RT'].values, index = df['Date'])
            time_RT.plot(figsize=(20,7), label='Retweets', legend=True)
            st.pyplot()

            # st.write(sns.jointplot(x=df['Retweets'], y=df['Likes'], data=df, kind='scatter'))
            # st.pyplot()
            # get the countPlot
            st.header("Count Plot for Different Sentiments")
            st.write(sns.countplot(df["Sentiment"]))
            st.pyplot()


        with col2:
            # Write Summary of the Tweets
            st.subheader("Summary of the Data")
            st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
            st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
            st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
            st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))

            # Piechart
            # if st.button("Get Pie Chart for Different Sentiments"):
            st.header("Generating A Pie Chart")
            a=len(df[df["Sentiment"]=="Positive"])
            b=len(df[df["Sentiment"]=="Negative"])
            c=len(df[df["Sentiment"]=="Neutral"])
            d=np.array([a,b,c])
            explode = (0.1, 0, 0)
            colors = ['#00FF00','#FF0000','#0000FF']
            st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"], colors=colors,autopct='%1.1f%%', startangle=90))
            st.pyplot()

            # Polarity plot
            st.header("Polarity Plot")
            st.write(sns.distplot(df['Polarity'], bins=30))
            st.pyplot()

            # get the countPlot Based on Verified and unverified Users
            st.header("Count Plot for Different Sentiments for Verified and unverified Users")
            st.write(sns.countplot(df["Sentiment"],hue=df.IsVerified))
            st.pyplot()


        col11, col12, col13 = st.beta_columns(3)
        with col11:

            # Create a Worlcloud

            st.header("WordCloud-ALL THINGS said about {}".format(Topic))
            text = " ".join(review for review in df.clean_tweet)
            stopwords = set(STOPWORDS)
            text_newALL = prepCloud(text,Topic)
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

        with col12:
            #Wordcloud for Positive tweets only
            st.header("WordCloud-POSTIVE THINGS about {}".format(Topic))
            text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_positive = prepCloud(text_positive,Topic)
            #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_positive)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

        with col13:
            # Wordcloud for Negative tweets only
            st.header("WordCloud-NEGATIVE THINGS about {}".format(Topic))
            text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_negative = prepCloud(text_negative,Topic)
            #text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_negative)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()





    st.sidebar.header("Brand Management and Social Listening")
    st.sidebar.info("dafjennvl ")
    st.sidebar.text("Built with Streamlit")

    #st.sidebar.subheader("Scatter-plot setup")
    #box1 = st.sidebar.selectbox(label= "X axis", options = numeric_columns)
    #box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
    #sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
    #st.pyplot()



    if st.button("Exit"):
        st.balloons()



if __name__ == '__main__':
    main()
