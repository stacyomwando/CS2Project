from imports import *

loaded_model=joblib.load("./pkl_objects/model.joblib", mmap_mode=None)
loaded_stop=joblib.load("./pkl_objects/stopwords.joblib", mmap_mode=None)
loaded_vec=joblib.load("./pkl_objects/vectorizer.joblib", mmap_mode=None)

app = Flask(__name__)

def classify(document):
 label = {-1.0: 'negative', 1.0: 'positive', 0.0:'neutral'}
 X = loaded_vec.transform([document])
 y = loaded_model.predict(X)[0]
 proba = np.max(loaded_model.predict_proba(X))
 return label[y], proba

class ReviewForm(Form):
 customerreviews = FileField('',[validators.DataRequired()])

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/forward', methods=['POST'])
# def move_forward():
#     return render_template('reviewform.html')

@app.route('/reviewinput', methods=['GET', 'POST'])
def reviewinput():
    if request.method == 'POST':
         form = ReviewForm(request.form)
    return redirect(url_for('results'))
        
    return render_template('reviewform.html', form=form)

@app.route(‘/results’, methods=[‘POST’])
def results():
 form = ReviewForm(request.form)
 if request.method == ‘POST’ and form.validate():
 review = request.form[‘moviereview’]
 y, proba = classify(review)
 return render_template(‘results.html’,content=review,prediction=y,probability=round(proba*100, 2))
 return render_template(‘reviewform.html’, form=form)


     
if __name__ == '__main__':
    app.run(debug=True)