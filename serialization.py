from aspect import *
from model import *
import joblib
from joblib import load, dump 
from nltk.corpus import stopwords

stopWords = set(stopwords.words("english"))
cv = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
classifier =SVC(C=1, kernel='linear', decision_function_shape='ovo', gamma='auto')
joblib.dump(stopWords,r'C:\Users\Stace Omwando\PycharmProjects\CS2\pkl_objects\stopwords.joblib')
joblib.dump(cv, r'C:\Users\Stace Omwando\PycharmProjects\CS2\pkl_objects\vectorizer.joblib')
joblib.dump(classifier, r'C:\Users\Stace Omwando\PycharmProjects\CS2\pkl_objects\model.joblib')
