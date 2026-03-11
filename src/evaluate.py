from sklearn.metrics import accuracy_score

def evaluate(model,X_val,y_val):

    pred = model.predict(X_val)

    acc = accuracy_score(y_val,pred)

    print("Accuracy:",acc)