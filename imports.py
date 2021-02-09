import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.externals
import pickle
import joblib
from joblib import dump, load
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from textblob import TextBlob
from flask import Flask, render_template, Response, request, redirect, url_for
from wtforms import Form, FileField, validators
import pickle
import sqlite3
import os
