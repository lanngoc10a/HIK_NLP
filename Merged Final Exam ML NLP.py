#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine 

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


# In[ ]:


df = pd.read_csv(r'C:\Users\sebas\Desktop\heart.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


X = df.loc[:,df.columns!="target"]
type(X)

y = df["target"]
type(y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_test_std


# In[ ]:


from sklearn import svm
clf_svm_1 = svm.SVC(kernel='linear', C=100)
clf_svm_1.fit(X_train_std, y_train)


# In[ ]:


y_train_pred = clf_svm_1.predict(X_train_std)
y_test_pred = clf_svm_1.predict(X_test_std)


# In[ ]:


y_test_pred


# ### Model Accuracy 

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[ ]:


confusion_matrix(y_test, y_test_pred)


# In[ ]:


CM = confusion_matrix(y_test, y_test_pred)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print(" TN " +str(TN), " TP " + str(TP), " FN "+ str(FN), " FP "+str(FP))


# In[ ]:


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print(TPR)


# In[ ]:


accuracy_score(y_test,y_test_pred)


# In[ ]:


print(classification_report(y_test, y_test_pred))


# In[ ]:


clf_svm_1.n_support_


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
fig, ax = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(clf_svm_1, X_test, y_test, normalize='true' ,cmap=plt.cm.Blues, ax=ax)


# In[ ]:


df.target.value_counts().plot.bar()


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'C':(0.1,1,100)}
clf_svm_1=svm.SVC(kernel='linear')
svm_grid_lin=GridSearchCV(clf_svm_1, params, n_jobs=-1,
                         cv=10, verbose=1, scoring='accuracy')


# In[ ]:


svm_grid_lin.fit(X_train_std, y_train)
svm_grid_lin.best_params_
linsvm_clf = svm_grid_lin.best_estimator_
accuracy_score(y_test, linsvm_clf.predict(X_test_std))


# In[ ]:


svm_grid_lin.best_params_


# ### CV poly and rbf 

# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'C':(0.1,1,10,25,50,100)}
clf_svm_p3=svm.SVC(kernel='poly')
svm_grid_lin=GridSearchCV(clf_svm_p3, params, n_jobs=-1,
                         cv=10, verbose=1, scoring='accuracy')


# In[ ]:


svm_grid_lin.fit(X_train_std, y_train)
svm_grid_lin.best_params_
linsvm_clf = svm_grid_lin.best_estimator_


# In[ ]:


svm_grid_lin.best_params_


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'C':(0.1, 0.2, 1,10,25,50,100)}
clf_svm_r=svm.SVC(kernel='rbf')
svm_grid_lin=GridSearchCV(clf_svm_r, params, n_jobs=-1,
                         cv=10, verbose=1, scoring='accuracy')


# In[ ]:


svm_grid_lin.fit(X_train_std, y_train)
svm_grid_lin.best_params_
linsvm_clf = svm_grid_lin.best_estimator_


# In[ ]:


svm_grid_lin.best_params_


# ### Polynomial

# In[ ]:


clf_svm_p3 = svm.SVC(kernel='poly', degree= 2, C= 1)
clf_svm_p3.fit(X_train_std, y_train)


# In[ ]:


y_train_pred = clf_svm_p3.predict(X_train_std)
y_test_pred = clf_svm_p3.predict(X_test_std)


# In[ ]:


accuracy_score(y_test,y_test_pred)


# In[ ]:


clf_svm_p3.n_support_


# ### RBF

# In[ ]:


clf_svm_r = svm.SVC(kernel='rbf', gamma=0.5, C=1)
clf_svm_r.fit(X_train_std, y_train)


# In[ ]:


y_train_pred = clf_svm_r.predict(X_train_std)
y_test_pred = clf_svm_r.predict(X_test_std)


# In[ ]:


accuracy_score(y_test, y_test_pred)


# In[ ]:


clf_svm_r.n_support_


# # NLP 

# In[1]:


import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #see below about "lemmatization"
lemmatizer = WordNetLemmatizer()
import re
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


df = pd.read_csv('Exam_MB210_NLP.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df["description"] = df["tagline"].astype(str) + " " + df["overview"].astype(str)


# In[6]:


df["description"].head(1)


# In[7]:


def preprocessing(corpus): #same as previous sessions
    stops = stopwords.words('english') #getting the english stop words list from NLTK
    corpusTokens = [word_tokenize(item) for item in corpus] #tokenizing with the word_tokenize method of NLTK
    def tokenCleaner(token): 
        return lemmatizer.lemmatize(re.sub('[^A-ZÆØÅa-zæøå0-9]','',token.strip().lower()))
    vocabulary = list() #initializing empty vocabulary as list
    for index,item in enumerate(corpusTokens): #looping over each document (which has been tokenized)
        documentTokens = [tokenCleaner(word) 
            for word in item if tokenCleaner(word) and word not in stops]  
        corpusTokens[index] = documentTokens  
        vocabulary += documentTokens  
    corpusNonstop = [(' ').join(document) for document in corpusTokens]  
    return corpusTokens, corpusNonstop, set(vocabulary)  


# In[8]:


pd.set_option("display.max_colwidth", None)
df['description'].head(1)


# In[9]:


corpusTokens, corpusNonstop, vocabulary = preprocessing(df.description) 


# In[10]:


print(corpusTokens[0],'\n\n',corpusNonstop[0], '\n\n', list(vocabulary)[:20]) #inspecting results


# In[11]:


len(list(set(word_tokenize((' ').join(corpusNonstop)))))


# In[12]:


df["corpusNonstop"] = corpusNonstop


# In[13]:


from collections import Counter
from datetime import datetime


# In[14]:


begin_time = datetime.now()


iteration = 0
divide = 1 #variable to vary the fraction of the dataset that is computed, i.e. when testing code

itermax = len(corpusTokens)//divide #the maximum number of iterations, function of 'divide'

count_all = Counter(corpusTokens[0]) #creating a counter, counting for first document

for i in range(1, itermax):
    count_all.update(Counter(corpusTokens[i])) #updating counter object with counts from all other docs.
    iteration+=1

countAllWords = count_all #storing the total counts in a second container (counter object)


#creating a DataFrame of the counter's top 20 words, setting column names and index
count_all = pd.DataFrame(count_all.most_common(20)).rename(columns={0:'word',1:'count'}).set_index('word')


end_time = datetime.now()
print('time spent',end_time-begin_time)
count_all.head(5)


# In[15]:


#barchart of the 20 most frequent words from the corpus, from above DataFrame
import matplotlib.pyplot as plt
import matplotlib.mathtext
plt.style.use('bmh')
plt.rcParams["font.family"] = "serif"

fig, most_common_words = plt.subplots(figsize=(12,3),dpi=300)

most_common_words.bar(count_all.index,count_all['count'],color='g')
most_common_words.set_xticks(count_all.index)
most_common_words.set_xticklabels(count_all.index, rotation='vertical')
most_common_words.set_ylabel('Occurrences')
most_common_words.set_xlabel('Words')
most_common_words.set_title("20 most frequent words in corpus")
plt.savefig('most_common_20.png')
plt.show()


# In[16]:


# Define helper functions
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''
    returns a tuple of the top n words in a sample and their 
    accompanying counts, given a CountVectorizer object and text sample
    '''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for 
             word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])


# In[ ]:





# # TF-IDF representation on description

# TF-IDF: term frequency times inverse document frequency. 
# 
# •Term frequencies are the counts of each word in a document
# 
# •Inverse document frequency means the inverse document frequency. The number of documents that contain that word goes up, the IDF (and hence the TF-IDF) for that word will go down

# ![tf_idf.PNG](attachment:tf_idf.PNG)

# idf (t,D) represents the idf for the term/word t in document D

# In[17]:


corpusRaw = df.description
corpusLabels = df.title


# In[18]:


def countDoc(document):
    from collections import Counter
    counts = Counter(document)
    return counts

def createBoW_tf(document, vocabulary):
    resultVector = []
    for word in vocabulary:
        if word in document:
            resultVector.append(countDoc(document)[word] / len(document))
        else:
            resultVector.append(0)
    return resultVector


# In[19]:


tfMatrix = []
for document in corpusTokens:
    tfMatrix.append(createBoW_tf(document, vocabulary))
#tfMatrix = np.array(tfMatrix)
tfMatrix = csr_matrix(tfMatrix)
print('shape',tfMatrix.shape)
from random import random
randomItem = int(random()*len(corpusRaw))
print('\nShowing vector for '+corpusLabels[randomItem]+'\n\n',np.round(tfMatrix[randomItem],3))


# In[20]:


#cosine similarity between these three documents, and scaling results between 0 and 1
pairwise_TF = tfMatrix*tfMatrix.T
pairwise_TF = MinMaxScaler().fit_transform(pairwise_TF.toarray().reshape(-1,1)).reshape(len(corpusRaw),len(corpusRaw))


# In[21]:


pd.DataFrame(pairwise_TF,columns=corpusLabels,index=corpusLabels)


# In[22]:


v_tr_tf = TfidfVectorizer(use_idf=True) #initializing the vectorizer
tf = v_tr_tf.fit_transform(corpusNonstop).toarray() #fit, transform corpus
vocabulary_trained = v_tr_tf.vocabulary_.keys() #get vocabulary from corpus


# In[23]:


from scipy.sparse import csr_matrix
tf = csr_matrix(tf)


# In[24]:


pairwise = tf*tf.T

for i in range(pairwise.shape[0]):
    pairwise[i,i] = 0


# In[25]:


source = 2
numberOfMatches = 2

results = np.argsort(pairwise.toarray()[source]).tolist()[-numberOfMatches:]
results.reverse()


print('query:')
print(corpusRaw[source])
print('----------')
print()
print('results:')
print()
for result in results:
    print(corpusRaw[result])
    print('--')


# In[26]:


pd.DataFrame((pairwise).toarray(),columns=corpusLabels,index=corpusLabels) #output pairwise similarity matrix


# In[27]:


corpusLabels


# In[28]:


v_tr_tfidf = TfidfVectorizer(min_df=1, use_idf=True, vocabulary=vocabulary_trained) #using IDF also with Scikit-learn
tfidf = v_tr_tfidf.fit_transform(corpusNonstop) 
pd.DataFrame((tfidf*tfidf.T).toarray(),columns=corpusLabels, index=corpusLabels)


# In[29]:


df[df.title=="Spider-Man"]


# In[30]:


queriesRaw = [ """ 'With great power comes great responsibility. """
               """After being bitten by a genetically altered spider, """
               """nerdy high school student Peter Parker is endowed with amazing powers' """  
             ]

queryLabels = ['Spider-Man']


# In[31]:


queriesNonstop = preprocessing(queriesRaw)[1] #preprocessing the queries


# In[32]:


queriesNonstop


# In[33]:


queriesNonstop = preprocessing(queriesRaw)[1] #preprocessing the queries

tfidfWithQueries = v_tr_tfidf.fit_transform(corpusNonstop)

Spider_Man_Similary_Df = pd.DataFrame((tfidfWithQueries*tfidfWithQueries.T).toarray(), 
             columns=corpusLabels, 
             index=corpusLabels)[queryLabels].drop(queryLabels)


# In[34]:


Spider_Man_Similary_Df.sort_values("Spider-Man", ascending = False).head(10)


# # LDiA

# •Compute the average position (centroid) of all the TF-IDF vectors 
# within the class (such as spam SMS messages).
# 
# •Compute the average position (centroid) of all the TF-IDF vectors 
# not in the class (such as nonspam SMS messages).
# 
# •Compute the vector difference between the centroids (the line 
# that connects them).

# In[35]:


# import sms-spam.csv
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD


# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=corpusNonstop).toarray()
print(tfidf_docs.shape)
print(type(tfidf_model.fit_transform(raw_documents=corpusNonstop)))


# In[37]:


print(tfidf_model.get_feature_names()[:20])


# In[38]:


print(tfidf_docs[:2])


# In[39]:


# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

n_samples = 4837
n_features = 1000
n_components = 10
n_top_words = 5
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)
tfidf = tfidf_vectorizer.fit_transform(df["corpusNonstop"])
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# In[40]:


import pandas as pd
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation


# In[41]:


# another implementation with more visualization

lda = LatentDirichletAllocation(
    n_components=n_components,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
)

lda.fit(tfidf)


# In[42]:


import matplotlib.pyplot as plt
# plot_top_words, see sklearn documentation: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    # plt.show()
    plt.savefig('topics.png')


# In[43]:


print(lda.components_)
get_ipython().run_line_magic('matplotlib', 'inline')
plot_top_words(lda, tfidf_feature_names, n_top_words, "Topics in LDA model")


# # Word2vec

# The parameters:
# 
# min_count = int - Ignores all words with total absolute frequency lower than this - (2, 100)
# 
# window = int - The maximum distance between the current and predicted word within a sentence. E.g. window words on the left and window words on the left of our target - (2, 10)
# 
# size = int - Dimensionality of the feature vectors. - (50, 300)
# 
# sample = float - The threshold for configuring which higher-frequency words are randomly downsampled. Highly influencial. - (0, 1e-5)
# 
# alpha = float - The initial learning rate - (0.01, 0.05)
# 
# min_alpha = float - Learning rate will linearly drop to min_alpha as training progresses. To set it: alpha - (min_alpha * epochs) ~ 0.00
# 
# negative = int - If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drown. If set to 0, no negative sampling is used. - (5, 20)
# 
# workers = int - Use these many worker threads to train the model (=faster training with multicore machines)

# Problem with this approach is that it assigned exactly one vector for each word, which is why it is considered as static word embeddings. This is particularly problematic when embedding words with multiple meaning (i.e. polysemous words), such as the word open; it can mean uncovered, honest, or available, depending on the context.

# In[44]:


from gensim.models import Word2Vec


# In[100]:


# Select features from original dataset to form a new dataframe 
df1 = df[['original_title','overview','tagline','title']]
# For each row, combine all the columns into one column
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)
# Store them in a pandas dataframe
df_clean = pd.DataFrame({'clean': df2})
sent = [row.split(',') for row in df_clean['clean']]
# show the example of list of list format of the custom corpus for gensim modeling 
sent[:2]


# In[101]:


model = Word2Vec(sent, min_count=1,workers=3, window =3, sg = 1)
model.wv.most_similar('Avatar')
model.wv.similarity('Avatar','Spider-Man 3')
vocab = model.wv.key_to_index
print(f'words in vocab: {len(vocab)}')
sim_words = model.wv.most_similar('Spider-Man', topn=7)
sim_words


# In[72]:


import numpy as np
#from scipy.stats import norm

def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model.wv[word]
    for item in target_list :
        try:
            if item != word :
                b = model.wv[item]
                cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
                cosine_dict[item] = cos_sim
        except:
            continue
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]


# In[76]:


# only get the unique tittle
title = list(df.title.unique()) 
#print(title)
# Show the most similar Mercedes-Benz SLK-Class by cosine distance 
cosine_distance(model,'Spider-Man 3',title,5)
 


# In[98]:


from sklearn.manifold import TSNE

def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
display_closestwords_tsnescatterplot(model, 'Spider-Man 3') 


# This T-SNE plot shows the top 10 similar movies to Spider-Man 3 in two-dimensional space.
# 

# In[ ]:




