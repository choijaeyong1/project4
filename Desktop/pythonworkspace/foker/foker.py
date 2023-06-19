import pandas as pd
train = pd.read_csv(r'C:\Users\ChoiJaeYong\Desktop\pythonworkspace\foker\poker_test_remix2.csv')
test = pd.read_csv(r'C:\Users\ChoiJaeYong\Desktop\pythonworkspace\foker\poker_train_remix2.csv')
train.drop(['A Hand', 'B First Rank', 'B First Suit', 'B Sixth Rank', 'B Sixth Suit', 'B Seventh Rank', 'B Seventh Suit',
            'C First Rank', 'C First Suit', 'C Sixth Rank', 'C Sixth Suit', 'C Seventh Rank', 'C Seventh Suit',
            'D First Rank', 'D First Suit', 'D Sixth Rank', 'D Sixth Suit', 'D Seventh Rank', 'D Seventh Suit',
            'E First Rank', 'E First Suit', 'E Sixth Rank', 'E Sixth Suit', 'E Seventh Rank', 'E Seventh Suit',
            'F First Rank', 'F First Suit', 'F Sixth Rank', 'F Sixth Suit', 'F Seventh Rank', 'F Seventh Suit',
            'G First Rank', 'G First Suit', 'G Sixth Rank', 'G Sixth Suit', 'G Seventh Rank', 'G Seventh Suit'], axis=1, inplace=True)
test.drop(['A Hand', 'B First Rank', 'B First Suit', 'B Sixth Rank', 'B Sixth Suit', 'B Seventh Rank', 'B Seventh Suit',
            'C First Rank', 'C First Suit', 'C Sixth Rank', 'C Sixth Suit', 'C Seventh Rank', 'C Seventh Suit',
            'D First Rank', 'D First Suit', 'D Sixth Rank', 'D Sixth Suit', 'D Seventh Rank', 'D Seventh Suit',
            'E First Rank', 'E First Suit', 'E Sixth Rank', 'E Sixth Suit', 'E Seventh Rank', 'E Seventh Suit',
            'F First Rank', 'F First Suit', 'F Sixth Rank', 'F Sixth Suit', 'F Seventh Rank', 'F Seventh Suit',
            'G First Rank', 'G First Suit', 'G Sixth Rank', 'G Sixth Suit', 'G Seventh Rank', 'G Seventh Suit'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action = 'ignore')

X_train = train.drop('G Hand', axis=1)
y_train = train['G Hand']
X_test = test.drop('G Hand', axis=1)
y_test = test['G Hand']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2, stratify = y_train)
from xgboost import XGBClassifier

model = XGBClassifier(
            n_jobs = -1,
            max_depth = 1,
            n_estimators = 100,
            learning_rate = 0.01,
            gamma = 0.1,
            random_state=2
            )
model.fit(X_train, y_train)
from sklearn.metrics import classification_report

print("학습셋 검증 정확도", model.score(X_train, y_train))
print("검증셋 검증 정확도", model.score(X_val, y_val))
y_pred_test = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_test)
print(accuracy)

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)