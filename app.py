from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


app = Flask(__name__)


df = pd.read_csv('cancer.csv')


selected_features = ['Air Pollution', 'Genetic Risk', 'Obesity', 'Balanced Diet', 'OccuPational Hazards', 'Coughing of Blood']
X = df[selected_features]


y = df['Level']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)


y_pred = rf_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy on Test Data: ", accuracy*100)
print("Confusion Matrix:\n", conf_matrix)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    air_pollution = float(request.form['air_pollution'])
    genetic_risk = float(request.form['genetic_risk'])
    obesity = float(request.form['obesity'])
    balanced_diet = float(request.form['balanced_diet'])
    occupational_hazards = float(request.form['occupational_hazards'])
    coughing_of_blood = float(request.form['coughing_of_blood'])


    input_data = [[air_pollution, genetic_risk, obesity, balanced_diet, occupational_hazards, coughing_of_blood]]

    
    prediction = rf_clf.predict(input_data)
    prediction_proba = rf_clf.predict_proba(input_data)


    predicted_class = prediction[0]
    predicted_prob = prediction_proba[0]


    result = {
        'predicted_class': predicted_class,
        'predicted_probability': dict(zip(rf_clf.classes_, predicted_prob))
    }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)



