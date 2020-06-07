
# Import Required Libraries (imp - LogisticRegression from sklearn for ML)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load the dataset using pandas & convert into numpy array
data = pd.read_csv("ML_Data.csv")
data = np.array(data)

# extract X (Probable Inputs) value and Y (Probable Outputs) value and convert into integer
X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
print(X,y)

# Split the dataset into Train (70%) and Test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

# Feed the Train dataset into LogisticRegression algorithm from sklearn library 
log_reg.fit(X_train, y_train)

# Use score method to get accuracy of model
score = log_reg.score(X_test, y_test)
print(score)

# Test out with your sample data and get the probablity
inputt=[int(x) for x in "50 70 800 5 10000 60".split(' ')]
final=[np.array(inputt)]
b = log_reg.predict_proba(final)
output='{0:.{1}f}'.format(b[0][1], 2)
print(output)

# Write (generate) your custom model in pickle file
pickle.dump(log_reg,open('model.pkl','wb'))

# Example: how to read the custom model (pickle file) form your actual application
model=pickle.load(open('model.pkl','rb'))


