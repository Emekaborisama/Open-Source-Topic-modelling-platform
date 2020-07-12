import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
#import seaborn as sns
cv = CountVectorizer()
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pickle

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Title

html_temp = """
<div style = "background.color:teal; padding:10px">
<h2 style = "color:white; text_align:center;"> Open Source Topic Modelling Platform</h2>
<p style = "color:white; text_align:center;"> Build a topic model with ease </p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

#st.cache()

#Dataset load 



text = st.text_input("", "Dataset")
st.write("Batch csv coming soon!!!!!")
if text == None:
    st.success("Pls upload your dataset")

df = {'text':[text]}#, 'name': [baba]}
df = pd.DataFrame(df)

#clean the dataset function
def clean_tweets(df, column):
    df.loc[:, column] = df[column].str.replace('^RT[\s]+', '')\
                                  .str.replace('https?:\/\/.*[\r\n]*', '')\
                                  .str.replace('[^\w\s]','')\
                                  .str.lower()
            
    return df


#df_clean = []
if st.button("Clean Dataset"):
    df = df.pipe(clean_tweets, 'text')
    st.success("Dataset has been cleaned, move to the next step")
    #df.append(df_clean)
    #df_clean = pd.DataFrame(df_clean)

df = df.pipe(clean_tweets, 'text')



#view clean data
html_temp = """
<div style = "background.color:teal; padding:10px">
<h4 style = "color:white;text_align:center;"> View Dataset </h4>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)
st.write("Display dataset")
if st.checkbox("View/Hide Dataset"):
    number = st.number_input("Number of rows to view", 1, 20)
    st.dataframe(df.head(number))


st.markdown("View columns")
#get only the columns 

if st.button("View/Hide columns in Dataset"):
    st.write(df.columns)



st.write("Shape of your dataset")
#shape of your dataset
if st.checkbox("View/Hide shape of your Dataset"):
    st.write(df.shape)

#drop unwanted columns
html_temp = """
<div style = "background.color:teal; padding:10px">
<h4 style = "color:white;text_align:center;"> Drop Unwanted Columns </h4>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)
st.markdown("Click on the button to drop unwanted columns")


if st.button("Drop unwanted columns"):
    df_select = df['text']
    st.dataframe(df.head(2))

df_select = df['text']



html_temp = """
<div style = "background.color:teal; padding:10px">
<h4 style = "color:white;text_align:center;"> Tokenization</h4>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

#Tokenize using CountVectorizer
tokens = ["CountVectorizer", "TFIDF Vectorizer", "Bert Tokenizer"]
select = st.radio("Select a Tokenizer", tokens)
if select == "CountVectorizer":
    count_data = cv.fit_transform(df_select)
    st.dataframe(count_data)
elif select == "TFIDF Vectorizer":
    st.markdown("_Coming soon_")
elif select == "Bert Tokenizer":
    st.markdown("_Coming soon_")

count_data = cv.fit_transform(df_select)

#train model
html_temp = """
<div style = "background.color:teal; padding:10px">
<h4 style = "color:white;text_align:center;"> Train your model with LDA</h4>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)
def helper(model, cv, n_top_words):
    st.markdown("_Model is training............._")
    words = cv.get_feature_names()
    st.markdown("_Training Completed............_")
    for topic_idx, topic in enumerate(model.components_):
        st.write("Topic:",topic_idx)
        st.write(" ".join([words[i]
                       for i in topic.argsort() [:-n_top_words - 1:-1]]))


number_topics = st.number_input("number of topics",1, 10)
number_words = st.number_input("number of words",1, 10)
#convert to int
#number_topics  = number_topics.astype(int)
#number_words = number_words.astype(int)
def build(text):
    lda2 = LDA(n_components = number_topics, n_jobs = 1)
    lda2 = lda2.fit(text)
    helper(lda2, cv, number_words)
    return helper



if st.button("Train Topic model"):
    output = build(count_data)


#Visualization
html_temp = """
<div style = "background.color:teal; padding:10px">
<h4 style = "color:white;text_align:center;"> Visualize Topics with LDA</h4>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

st.markdown("_coming soon_")



    
   





















