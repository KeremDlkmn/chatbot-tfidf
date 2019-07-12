## İmport Libraries
import nltk         # nltk
import numpy as np  # numpy
import random       # random
import string       # string
from sklearn.feature_extraction.text import TfidfVectorizer # Benzerlik Matrisi İçin
from sklearn.metrics.pairwise import cosine_similarity      # Cos Benzerliği İçin

## ----- Corpus okuma işemleri başlangıç ----- ##
f = open('chatbot.txt','r',errors='ignore') # errors='ignore' hataları yok sayar
readRaw = f.read() # corpus'u okuduk

readRaw = readRaw.lower() # İlk adım kelimeleri küçük hale getirdik

sent_tokens = nltk.sent_tokenize(readRaw)# okuduğumuz corpus'un cümle listelerini oluşturur
word_tokens = nltk.word_tokenize(readRaw)# okuduğumuz corpus'un kelime listelerini oluşturur

## ----- Corpus okuma işlemleri bitti ----- ##

## ----- Lemmatization işlemi başlangıç ----- ##
lemmer = nltk.stem.WordNetLemmatizer() # WordNet sözlüğünden kelimeleri aldım

# LemTokens: Parametre olarak kelime(tokens) vereceğiz.
# LemTokens bu verilen kelimenin kökünü bulur lemmatize sayesinde
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# .,! gibi işaretlerin unicode değerlerini bir dict haline getirdik. Böylece ilerleyen zamanlarda bu işaretleri kaldırmak istersek bu dict'i kullanabilirim
# string kütüphanesi içerisinde bulunan translate fonksiyonu unicode değerler ile çalışır. o yüzden dict haline getirirken unicode değerlerini aldık
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# LemNormalize: Parametre olarak bir text alır
# İlk başta bu text küçük harflere çevrilir ve translate fonksiyonu ile sözlük içerisindeki karakterler var ise silinir
# text word_tokenize sayesinde kelimelere ayrılır ve LemTokens fonksiyonuna yollanarak kelimenin kökü bulunur.
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

## ----- Lemmatization işlemi bitiş ----- ##

## ----- Botumuza seslenildiğinde vereceği cevap başlamgıç ----- ##
INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

# greeting(Karşılama): bir cümle alacağız, bu cümleyi split
# edeceğiz içerisinde INPUT sabitlerinin değerlerinden biri varsa
# RESPONSES içerisinden rasgele bir veri seçip yollayacağız
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in INPUTS:
            return random.choice(RESPONSES)


## ----- Yanıt Üretme Başlangıç ----- ##
def response(user_response):
    robo_response=''            # chatbot'un vereceği cevap için boş bir değişken tanımladım
    sent_tokens.append(user_response) # sent_tokens'a user'ın sorusunu ekledim

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') # Oluşturulacak vektör LemNormalize fonksiyonunu kullanacak ve stop words olarak english stop words leri kullanacak
    tfidf = TfidfVec.fit_transform(sent_tokens) # Bag Of Words mantığında ki gibi bir matris döndürür. feature adları kelimeler(sütun) corpus'ta ağırlıklı olan kelimeler olur
    vals = cosine_similarity(tfidf[-1], tfidf)  # Kosinüs benzerliğini bulmamızı sağlar. tfdif[-1] kullanıcının cümlesi
    idx=vals.argsort()[0][-2] # argsort numpy'ın sıralama fonksiyonudur [-2] dönecek olan cümledir
    flat = vals.flatten() # matris tek boyuta iner
    flat.sort()           # sıralanır
    req_tfidf = flat[-2]  # kosinüs derecesini aldık
    if(req_tfidf==0):     # kosinüs derecesi 0 ise hata verdir.
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else: # değilse o benzerliği olan cevabı robo response'a ekle
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True  # Durmadan while içerisinde kalmamız için oluşturulmuş bir değişkendir.
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()             # dışarıdan user text girdi
    user_response=user_response.lower() # girilen text küçük harflere dönüştürüldü
    if(user_response!='bye'):           # girilen text bye değilse
        if(user_response=='thanks' or user_response=='thank you' ): # girilen text teşekkür ederim ise chatbot rica ederim der ve kendini kapatır
            flag=False # kendini kapatmak için flag değişkeni false olur
            print("ROBO: You are welcome..") # chatbot rica eder
        else:
            if(greeting(user_response)!=None):  # teşekkür etmediyse girilen text hi, hello gibi oto cevap verilecek olan bir kelime mi?
                print("ROBO: "+greeting(user_response)) # eğer öyleyse random response dönecek olan fonksiyon çalışsın
            else: # değilse
                print("ROBO: ",end="")
                print(response(user_response)) # kelimeler arasındaki benzerlikleri ölçeceğimiz fonksiyona git.
                sent_tokens.remove(user_response) # en sonda eklenen kelimeyi çıkaralım böylece her soruyu ekleyerek gitmesin sadece o anda sorulan soru eklensin ki ona göre benzerlik bulalım
    else:
        flag=False
        print("ROBO: Bye! take care..")
