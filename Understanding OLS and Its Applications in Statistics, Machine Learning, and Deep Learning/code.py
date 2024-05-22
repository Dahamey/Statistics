import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Example dataset
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3*X.squeeze() + 2 + np.random.randn(100) * 0.5

# Add a constant (intercept) term to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Plot the data and the regression line
plt.scatter(X[:, 1], y, label='Data')
plt.plot(X[:, 1], model.predict(X), color='red', label='OLS Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# ## Sum of Squared Residuals (SSR)

# In[2]:


# Calculate residuals
residuals = y - model.predict(X)

# Compute sum of squared residuals
ssr = np.sum(residuals**2)
print(f'Sum of Squared Residuals (SSR): {ssr}')


# ## Regression Analysis

# In[3]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

# Example dataset
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5, 6, 7, 8],
    'x2': [2, 3, 4, 5, 6, 7, 8, 9],
    'y': [1, 2, 1.3, 3.75, 2.25, 2.8, 3.2, 4.5]
})

X = data[['x1', 'x2']]
y = data['y']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary without convergence warning
print(model.summary())


# ## OLS in Machine Learning : Feature Selection

# In[4]:


import seaborn as sns

# Example dataset with multiple features
np.random.seed(0)
X = np.random.rand(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] + 1*X[:, 2] + np.random.randn(100) * 0.5

# Add a constant (intercept) term to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Feature importance (coefficients)
coefficients = model.params
print(f'Feature coefficients: {coefficients}')

# Visualize feature importance
sns.barplot(x=['Intercept', 'Feature1', 'Feature2', 'Feature3'], y=coefficients)
plt.title('Feature Importance')
plt.show()


# ## Model Evaluation

# In[5]:


# R-squared value
r_squared = model.rsquared
print(f'R-squared: {r_squared}')   # R-squared: 0.8448379108798427


# ##  Deep Learning : Loss Function and Gradient Descent

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense 
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(1000, 1)
y = 3*X.squeeze() + 2 + np.random.randn(1000) * 0.5

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build a simple neural network
model = Sequential([
    Input(shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1)
])

# Compile the model with Mean Squared Error loss
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss (MSE): {loss}')
