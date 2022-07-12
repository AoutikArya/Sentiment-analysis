'''System Module.'''
import re
import time
import streamlit as st
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import keras
from tf.keras.preprocessing.text import one_hot
from tf.keras.preprocessing.sequence import pad_sequences
from PIL import Image
VOC=1000
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
image = Image.open('download.png')
st.image(image, width=200)
st.header("Sentiment Analysis")
with st.spinner('loading model into memory:...'):
    model= tf.keras.models.load_model('./models/')
sentence=[st.text_input("Enter your sentence here: ")]
data=pd.DataFrame(data=sentence,columns=['tweet'])
x=data['tweet']
ps=PorterStemmer()
wl=WordNetLemmatizer()
k=[]
for i in range(len(data)):
    REVIEW=re.sub('[^a-zA-Z]',' ',str(x[i]))
    REVIEW=REVIEW.split()
    REVIEW=[wl.lemmatize(word) for word in REVIEW if word not in stopwords.words()]
    REVIEW=" ".join(REVIEW)
    k.append(REVIEW)
oh=[one_hot(word,VOC)for word in k]
pad=pad_sequences(oh,padding='pre',maxlen=50)
X=pad
if st.button("Analyse"):
    with st.spinner("Analysing.."):
        time.sleep(2)
        pred=(model.predict(X)>0.5).astype('int32')[0][0]
        if pred==0:
            st.write('This tweet is not racist/sexist.')
            st.balloons()
        else:
            st.write('This tweet is racist/sexist.')
