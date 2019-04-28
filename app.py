from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection  import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv('spam.csv',encoding='latin-1')
    data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

    data['label'] = data['v1'].map({'ham':0, 'spam':1})
    X = data['v2']
    y = data['label']

    cv = CountVectorizer()   # 将文本中的词语转换为词频矩阵
    X = cv.fit_transform(X)   # 计算各个词语出现的次数

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # persist model in a standard format
    # persist the model for future use without having to retrain
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_predict = clf.predict(vect)

    return render_template('result.html', prediction = my_predict)


if __name__ == '__main__':
    app.run(debug=True)

