## İmport Libraries
import nltk         # nltk
import numpy as np  # numpy
import random       # random
import string       # string
from sklearn.feature_extraction.text import TfidfVectorizer # Benzerlik Matrisi İçin
from sklearn.metrics.pairwise import cosine_similarity      # Cos Benzerliği İçin

## ----- Start Read a Corpus File ----- ##
f = open('chatbot.txt','r',errors='ignore') # errors='ignore' hataları yok sayar
readRaw = f.read() # corpus'u okuduk

readRaw = readRaw.lower() # İlk adım kelimeleri küçük hale getirdik

sent_tokens = nltk.sent_tokenize(readRaw)# okuduğumuz corpus'un cümle listelerini oluşturur
word_tokens = nltk.word_tokenize(readRaw)# okuduğumuz corpus'un kelime listelerini oluşturur

## ----- End Read a Corpus File ----- ##

## ----- Start Operation of Lemmatization ----- ##
lemmer = nltk.stem.WordNetLemmatizer() # WordNet sözlüğünden kelimeleri aldım

# LemTokens Finds The Root of The Word
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

## ----- End Operation of Lemmatization ----- ##

INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in INPUTS:
            return random.choice(RESPONSES)


def response(user_response):
    robo_response=''           
    sent_tokens.append(user_response) 

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf) 
    idx=vals.argsort()[0][-2] 
    flat = vals.flatten() 
    flat.sort()          
    req_tfidf = flat[-2]  
    if(req_tfidf==0):    
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else: 
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True 
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()             
    user_response=user_response.lower()
    if(user_response!='bye'):           
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None): 
                print("ROBO: "+greeting(user_response)) 
            else: # değilse
                print("ROBO: ",end="")
                print(response(user_response)) 
                sent_tokens.remove(user_response) 
    else:
        flag=False
        print("ROBO: Bye! take care..")
