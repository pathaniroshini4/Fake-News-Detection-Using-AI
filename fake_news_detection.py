import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'fake-and-real-news-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F4831777%2F8165591%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240718%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240718T064854Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D39ad614b72006431b9d2ae380bea8eaf5d68ea80ca99bfe5a2a5c29ae5aed80ea373e8eccdd83bb1f3655abd7a0ab5913d16296906c0e7cef1e128ad0df547b00e919f0ced879a7775bd36c4715aa3230858eca19869a74dda7330fc5134d6962bf2b9323c6a4a1e209fb7e1a28bd020a70a1faa8898c8daf6dec851d583eb92bbb5141cd6d91345efc89b206cb343e3900fda7ee9aaddadc499e4a0dc8639d5efcfd43592060e93af75db933b43f7cfe53bff237e8e790741cc8a41ea11504de08ebe26972864b1553ee5ca2abefb44830805e98c18df65909312b9945378c2974c0d158f8a9e23af128b2a65c5c8d54e2f2ebcfa5ce87f253c3952d0afe6f9'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import pandas as pd
import numpy as np

data_true=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
data_false=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

data_true['label']=1
data_false['label']=0

data_true.duplicated().sum()
data_false.duplicated().sum()

data_true.drop_duplicates(inplace=True)
data_false.drop_duplicates(inplace=True)

data_true.columns

data_true[data_true['title']==None].shape[0]
data_false[data_false['title']==None].shape[0]

"""### Preprocessing"""

import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import spacy

nltk.download('stopwords')
Stop_words= stopwords.words('english')
model=spacy.load('en_core_web_sm')

def lemmatize(text,model):
    doc=model(text)
    lemmatized_tokens=[i.lemma_ for i in doc if i.is_punct==False and i.is_stop==False]
    return ' '.join(lemmatized_tokens)

### Exploring the data
data_true.head()

###Now concatinate the two data frames
data=pd.concat([data_true,data_false],axis=0)
data.head()

### Now provide the new indexes
data.shape

import random
random.seed(42)

Index=np.arange(0,44689,1)
random.shuffle(Index)

data.index=Index

data.sort_index(inplace=True)

### there are duplicate rows present in the data so we habe to handle them first
data.drop_duplicates('text',inplace=True)
data['text'].duplicated().sum()

data.drop_duplicates('title',inplace=True)

### Now lets remove the subject and data as they are not relvent to the truth value of the news
data.columns

data.drop(['subject','date'],axis=1,inplace=True)

### we would only try to take a small sample of the bigger subject because of the limited resources
data_sample=data.iloc[1:1000,:]

data_sample.shape

import warnings
warnings.filterwarnings('ignore')

data_sample['text']=data_sample['text'].apply(lambda x:lemmatize(x,model))

from imblearn.over_sampling import RandomOverSampler

oversampler=RandomOverSampler()

X=data_sample['text'].values
y=data_sample['label'].values

X_resample,y_resample=oversampler.fit_resample(X.reshape(-1,1),y)

data_sample['label'].value_counts()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score


X_resample=X_resample.reshape(-1)

X_train,X_test,y_train,y_test=train_test_split(X_resample,y_resample,train_size=0.8,random_state=42,stratify=y_resample)

clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('Learning Model',LogisticRegression())
])

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))