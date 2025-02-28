
# Step 3: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Step 4: Load both datasets
true_df = pd.read_csv(r'C:/Users/Samarpita Roy/OneDrive/Desktop/FakeNewsDetection/FakeNewsDetection/News _dataset/True.csv')
true_df['label'] = 0  # Add a label column for true news

fake_df = pd.read_csv(r'C:/Users/Samarpita Roy/OneDrive/Desktop/FakeNewsDetection/FakeNewsDetection/News _dataset/Fake.csv')
fake_df['label'] = 1  # Add a label column for fake news


# Step 5: Combine both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 7: Train the Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Step 8: Make predictions and Evaluate the model
# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

# Step 9: Display results
# Display the result
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report_result)

# Step 10: User Input
# Take input from the user
user_input = input("Enter a news article: ")

# Make a prediction
prediction = model.predict([user_input])

# Display the result
if prediction[0] == 0:
    print("The news is likely to be true.")
else:
    print("The news is likely to be fake.")
