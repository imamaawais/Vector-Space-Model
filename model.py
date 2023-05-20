#libraries and utilities
#tokenization
import nltk
from nltk.tokenize import word_tokenize

import re #remove special characters

#stemming
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

#stop words
file = open('Stopword-List.txt','r')
stopWords = file.read() #read and store stop word list
file.close()

#vectors
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

#saving inverted index
import json

#to get execution time
import time

#GUI
import tkinter as tk

import numpy as np

####################################################################################
#preprocessing
####################################################################################

def stemmer(tokens):
    #make object
    stemmer = PorterStemmer()

    #stem words
    stemTokens = [stemmer.stem(token) for token in tokens]

    return stemTokens


#remove stop words
def removeStopWords(tokens):
    #create an empty list to store filtered tokens
    filTokens = []
    
    for token in tokens:
        if token not in stopWords:
            filTokens.append(token)

    return filTokens
    
def caseFolding(tokens):
    cfTokens = [token.lower() for token in tokens]
    return cfTokens

#remove special characters
def removeChar(tokens):

    #consider only alphabets in english
    pattern = re.compile('[^a-zA-Z]')
    #create empty list to store cleaned tokens
    newTokens = [] 

    #iterate through list
    for token in tokens:
        newToken = pattern.sub('', token) #clean the string
        newTokens.append(newToken) #append to the new  cleaned tokens list

    return newTokens

#extracting features
def tokenizer(content):

    #tokenize using nltk
    content = content.replace("-",'1 ').replace(';',' ')
    tokens = word_tokenize(content)
    #newtokens = word_tokenize(content)
    
    #return the tokens
    return tokens
        
def normalization(contents):
    
    #tokenize the content of the file
    tokens = tokenizer(contents)
    #print(tokens)
    #remove the special characters
    clean = removeChar(tokens)
    #print(clean)
    #print(len(clean))
    #change all terms to lowercase
    caseFolded = caseFolding(clean)
    #print(caseFolded)
    #filter the terms to remove stop words
    filtered = removeStopWords(caseFolded)
    #print(filtered)
    stemmed = stemmer(filtered)
    #return the final tokens
    return stemmed



####################################################################################
#instantiation
####################################################################################

#Reading documents
def readDoc(docNo):
    
    #create a list that contains the content of the file
    
    file = open('Dataset/'+str(docNo)+'.txt','r')
    #read content of file
    content = file.read()
    #close file
    file.close()
    #return the obtained contents of the files
    return content

def saveVocabList(vocab):
    #convert array into dictionary to save in json file
    vocabList = vocab.tolist()

    with open("vocabList.json", "w") as file:
        json.dump(vocabList, file)

def loadVocabList():
    with open("vocabList.json", "r") as file:
        vocabList= json.load(file)

    #convert the dictionary into a list first and then into an array
    vocab = np.array(vocabList)
    return vocab
    
def saveDocVectors(vectors):
    vectorList = vectors.toarray().tolist()
    with open("docVectors.json", "w") as file:
        json.dump(vectorList, file)

def loadDocVectors():
    with open("docVectors.json", "r") as file:
        vectorList = json.load(file)
    
    vectors = np.array(vectorList)
    return vectors    


def docVectorizer():
    tempCorpus = []
    #iterate over all 30 docs
    for docNo in range (1,31,1):
        #read the doc
        contents = readDoc(docNo)
        
        #tokenize and normalize the contents
        preprocessed = normalization(contents)

        #create corpus by appending list of tokens
        tempCorpus.append(preprocessed)
    
    #form corpus in form of set of terms for every doc because fit_transform uses list of strings
    corpus = [' '.join(tokens) for tokens in tempCorpus]
    #for i in corpus:
    #   print(i,"\n\n\n")

    #instantiate vectorizer and sort=True for lexical ordering
    vectorizer = TfidfVectorizer(max_features=1550)
    vectors = vectorizer.fit_transform(corpus)

    print(vectors)

    #normalize the document vectors
    normVectors = normalize(vectors, norm='l2', axis=1) #l2 shows ucilidean normalization, axis=1 shows rows norm

    #save normalized vocab list to process later
    saveVocabList(vectorizer.get_feature_names_out())
    #print(len(vectorizer.get_feature_names_out()))

    #save normalized document vectors
    saveDocVectors(normVectors)

####################################################################################
#Query processing
####################################################################################
def queryVectorizer(query):

    #normalize the query
    queryTokens = normalization(query)
    #convert into string
    queryString = ' '.join(queryTokens)

    #load the vocab list
    vocab = loadVocabList()

    #instantiate vectorizer and give vocab made pereviously
    vectorizer = TfidfVectorizer(vocabulary=vocab,sublinear_tf=True)
    vectors = vectorizer.fit_transform([queryString])

    #normalize the document vectors
    normVectors = normalize(vectors, norm='l2', axis=1) #l2 shows ucilidean normalization, axis=1 shows rows norm

    return normVectors.toarray()
    #return vectors.toarray()
    

def similarity(queryVector):
    #load the document vectors
    docVectors = loadDocVectors()
    
    #initialize empty list to store scores
    scores = []

    #iterate over all the docs and calculate dot product for each doc,query pair
    for i in range(0,30,1):
        #only calculate dot product for the common terms in q and d[i]
        score = np.dot(queryVector,docVectors[i])
        scores.append((i+1, score))

    scores = sorted(scores, key=lambda x: x[1],reverse=True)

    return scores



####################################################################################
#Driver code
####################################################################################

def driver():
    #call document vectorizer to make documents vectors
    docVectorizer()

    #take query input
    #query = input("enter query: ")
    query = queryEntry.get()

    # Start the timer
    startTime = time.time()

    #make query vector
    queryVector = queryVectorizer(query)

    #calculate similarity
    scores = similarity(queryVector)

    # End the timer
    endTime = time.time()
    
    # Calculate the elapsed time
    elapsedTime = (endTime - startTime) * 1000 # in milliseconds

    result = [ ]
    for i in range(0,30):
        if scores[i][1]>0.005:
           result.append(scores[i][0])

    # Update the result label
    if len(result) == 0:
        resultLabel.config(text="No results found")
    else:
        resultLabel.config(text=result)

    # Update the time label
    timeLabel.config(text=f"Elapsed time: {elapsedTime:.2f} milliseconds")
    



def clear():
    # Clear the query entry widget
    queryEntry.delete(0, tk.END)
    
    # Clear the result label
    resultLabel.config(text="")
    
    # Clear the time label
    timeLabel.config(text="")
    queryEntry


if __name__ == "__main__":

    # Create the main window
    root = tk.Tk()
    root.geometry("400x300+100+100")
    root.title("Boolean Retrieval Model")
    
    # Create the query label and entry widget
    queryLabel = tk.Label(root, text="Enter Query:")
    queryEntry = tk.Entry(root, width=50)
    
    # Create the search button
    searchButton = tk.Button(root, text="Search", command=driver)
    
    # Create the clear button
    clearButton = tk.Button(root, text="Clear", command=clear)
    
    # Create the result label and time label
    resultLabel = tk.Label(root, text="")
    timeLabel = tk.Label(root, text="")
    
    # Add padding
    queryLabel.pack(pady=5)
    queryEntry.pack(pady=5)
    searchButton.pack(pady=5)
    clearButton.pack(pady=5)
    resultLabel.pack(pady=5)
    timeLabel.pack(pady=5)
    
    # Pack the widgets
    queryLabel.pack()
    queryEntry.pack()
    searchButton.pack()
    clearButton.pack()
    resultLabel.pack()
    timeLabel.pack()


    root.config(bg='black')

    queryLabel.config(fg='white', bg='black')
    queryEntry.config(fg='black', bg='white')

    searchButton.config(fg='white', bg='black')
    clearButton.config(fg='white', bg='black')

    resultLabel.config(fg='white', bg='black')
    timeLabel.config(fg='white', bg='black')

    
    # Start the main event loop
    root.mainloop()

