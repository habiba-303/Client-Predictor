import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from dataprocessing import build_preprocessor, load_data, split_xy


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------- 1) data ----------
df = load_data()
X, y = split_xy(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#------------------------------

models = [
    {
        "model" : [LogisticRegression()],
        "model__max_iter" : [1000],
        "model__class_weight" : ["balanced", None]
    },

    {
        "model" : [KNeighborsClassifier()],
        "model__n_neighbors" : [5, 10, 15, 20],
        "model__weights" : ["uniform", "distance"],
        "model__leaf_size" : [10, 20, 30],
        "model__metric" : ["manhatten", "euclidean", "minkowski"],
    },

    {
        "model" : [DecisionTreeClassifier()],
        "model__criterion" : ["gini", "entropy"],
        "model__splitter" : ["best", "random"],
        "model__max_depth" : [3, 5, 10, 15],
        "model__min_samples_split" : [2, 3, 4, 5],
        "model__min_samples_leaf" : [2, 3, 4, 5],
        "model__max_features" : ["sqrt", "log2", None],
        "model__class_weight" : ["balanced", None]
    },

    {
        "model" : [RandomForestClassifier()],
        "model__n_estimators" : [100, 200, 300, 400, 500],
        "model__max_depth" : [5, 10, 15, 20],
        "model__min_samples_split" : [2, 3, 4, 5],
        "model__max_features" : ["sqrt", "log2", None],
        "model__criterion" : ["gini", "entropy"],
        "model__class_weight" : ["balanced", None]

    },

    {
        "model" : [GaussianNB()]
    },

    {
        "model" : [SVC()],
        "model__C" : [0.01, 0.1, 0, 1],
        "model__gamma" : ["scale", "auto"],
        "model__class_weight" : ["balanced", None], 
        "model__kernel" : ["linear", "poly", "rbf"],
        "model__probability" : [True] 
    }
]
# ---------- 2) Preprocessor & Training Loop ----------

preprocessor = build_preprocessor()

for model_dict in models:

    clf = model_dict["model"][0]  # clf(classifier) => extacting/ looping on every model from index 0
    model_name = type(clf).__name__
    
    param_dist = {key : value for key , value in model_dict.items() if key != "model"}  # key => name of parameter , ex: "model__max_iter" || value , ex: [1000]
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    
    print(f"===Training & Tuning: {model_name}===")
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,          # no. of iterations
        cv=5,               # Cross-validation folds => we split the data into folds (5) ,
                            # train it with 4 and test it with the remainig one, then we take the average of 5 folds
        scoring='accuracy',      
        random_state=42,
        n_jobs=-1
    )
    
    # training
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # testing
    y_pred = best_model.predict(X_test)

    
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    
    # results
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Best Params: {search.best_params_}\n")
    
    # save the model
    joblib.dump(best_model, MODELS_DIR / f"{model_name.lower()}_model.pkl")




