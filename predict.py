import pickle


def cd_detection(features):
    pickled_model = pickle.load(open('churn_detection.pkl', 'rb'))
    churn = str(round(list(pickled_model.predict([features]))[0]))

    return str("churn detection " + churn)
test_features=[1.0,
 1.0,
 1.0,
 0.0,
 29.0,
 1.0,
 2.0,
 1.0,
 0.0,
 0.0,
 2.0,
 0.0,
 2.0,
 2.0,
 0.0,
 1.0,
 2.0,
 98.5,
 3004.15]
cd_detection(test_features)