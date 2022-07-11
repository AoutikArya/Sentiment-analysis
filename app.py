'''System Module.'''
import streamlit as st
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import time
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
voc=1000
nltk.download('stopwords')
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
    review=re.sub('[^a-zA-Z]',' ',str(x[i]))
    review=review.split()
    review=[wl.lemmatize(word) for word in review if word not in stopwords.words()]
    review=" ".join(review)
    k.append(review)
oh=[one_hot(word,voc)for word in k] 
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
 


