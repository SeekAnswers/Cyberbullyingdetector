# %% [markdown]
# Firstly, we would be importing the needed libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer

from time import time
import pickle

# %% [markdown]
# Let's get the dataset in(Import the dataset)

# %%
df = pd.read_csv(r'C:\Users\kccha\OneDrive\Desktop\Programming\Cyber bullying detection\dataset.csv')

# %% [markdown]
# Lets assess the first five stats

# %%
df.head()

# %% [markdown]
# Let's check the labels

# %%
df['label'].unique()

# %% [markdown]
# We have to do dataframe manipulation to convert -1 to zero/1 so we can "usefulize" the dataset

# %%
def perform_data_manipulation():
    df = pd.read_csv(r'C:\Users\kccha\OneDrive\Desktop\Programming\Cyber bullying detection\dataset.csv')
    for index in df.index:
        if df.loc[index, 'label']==-1:
            df.loc[index, 'label'] = 1
    return df

# %% [markdown]
# We call the method

# %%
df = perform_data_manipulation()

# %% [markdown]
# Let's confirm the conversion

# %%
df.head()

# %%
df['label'].unique()

# %%
df.shape[0]

# %% [markdown]
# Let's check how many values are 1 and how many are 0, this tells us if the data is balanced or imbalanced

# %% [markdown]
# So we can do this classification using a Pie chart

# %%
def performdatadistribution(df):
    total = df.shape[0]
    num_non_toxic = df[df['label']==0].shape[0]
    
    slices = [num_non_toxic/total, (total-num_non_toxic)/total]
    
    labeling = ['Non-Toxic', 'Toxic']
    
    explode = [0.2, 0]
    
    plt.pie(slices, explode = explode, shadow=True, autopct="%1.1f%%", labels = labeling, wedgeprops={'edgecolor': 'black'})
    
    plt.title('Number of Toxic Vs Non- Toxic Test Sample')
    
    plt.tight_layout()
    
    plt.show()

# %%
performdatadistribution(df)

# %% [markdown]
# Let's try to remove the patterns

# %%
def remove_pattern(input_txt, pattern):
    if (type(input_txt)==str):
        r = re.finall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt
    else:
        return '' 

# %%
df.head(1)

# %%
# import pandas as pd
# import numpy as np
# import nltk

# Ensure the necessary NLTK resources are available
nltk.download('wordnet')

# Function to remove a specific pattern from the text (e.g., @user mentions)
def remove_pattern(input_txt, pattern):
    import re
    return re.sub(pattern, '', input_txt)  # Correct: Defined remove_pattern function

def datasetCleaning(df):
    # Calculate the length of the headline
    df['length_headline'] = df['headline'].str.len()
    
    # Concatenate the DataFrame with itself
    combined_df = pd.concat([df, df], ignore_index=True)  # Correct: Used pd.concat([df, df], ignore_index=True)
    
    # Remove @user mentions
    combined_df['tidy_tweet'] = np.vectorize(remove_pattern)(combined_df['headline'], '@[\w]*')

    # Remove non-alphabetic characters and # symbols
    combined_df['tidy_tweet'] = combined_df['tidy_tweet'].str.replace('[^a-zA-Z#]', ' ', regex=True)
    # Correct: Added regex=True in str.replace to ensure the method treats the pattern as a regular expression

    # Remove words with less than 3 characters
    combined_df['tidy_tweet'] = combined_df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    # Calculate the length of the cleaned tweet
    combined_df['length_tidy_tweet'] = combined_df['tidy_tweet'].str.len()
    # Correct: Replaced ['tidy_tweet'].str.len() with combined_df['tidy_tweet'].str.len()

    # Tokenization
    tokenized_tweet = combined_df['tidy_tweet'].apply(lambda x: x.split())

    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    # Correct: Changed lemmatizer.lemmatizer(i) to lemmatizer.lemmatize(i)

    # Rejoin the tokens into a single string
    combined_df['tidy_tweet'] = tokenized_tweet.apply(lambda x: ' '.join(x))
    # Correct: Added logic to rejoin tokens into a single string after lemmatization

    return combined_df, df


# %%
combined_df, df = datasetCleaning(df)

# %% [markdown]
# Let's split the Dataset

# %%
from sklearn.model_selection import train_test_split

# %%
combined_df.head()

# %%
def performdatasplit(x, y, combined_df, df):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_df['tidy_tweet'], combined_df['label'], test_size=x, random_state=y)

    # Load stopwords from a file
    with open('stopwords.txt', 'r') as file:
        content = file.read()
    content_list = content.split('\n')

    # Initialize TF-IDF Vectorizer with the stopwords
    Tfidvector = TfidfVectorizer(stop_words=content_list, lowercase=True)

    # Transform the training data into TF-IDF features
    training_data = Tfidvector.fit_transform(X_train.values.astype('U'))

    # Transform the testing data into TF-IDF features
    testing_data = Tfidvector.transform(X_test.values.astype('U'))

    # Save the entire TF-IDF vectorizer object
    filename = 'tfidfvectorizer.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(Tfidvector, file)

    # Return the split datasets and other relevant objects
    return X_train, X_test, y_train, y_test, testing_data, filename, training_data, content_list


# %%
X_train , X_test, y_train, y_test, testing_data, filename, training_data, content_list = performdatasplit(0.2, 42, combined_df, df)

# %% [markdown]
# Apply Machine learning algorithms

# %%
!pip install xgboost
import xgboost as xgb
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from time import time


# %%
def pipeline(X_train, y_train, X_test, y_test):
    MODELS = [
        LinearSVC(),
        LogisticRegression(),
        MultinomialNB(),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        SGDClassifier()
    ]

    final_result = []

    for model in MODELS:
        results = {}  # Initialize the results dictionary inside the loop
        results['Algorithm'] = model.__class__.__name__

        start = time()
        print(f'Training {model.__class__.__name__}...')
        
        # Fit the model
        model.fit(X_train, y_train)
        end = time()
        
        results['Training Time'] = end - start

        # Save the model to a file
        filename = model.__class__.__name__ + '.pkl'
        pickle.dump(model, open(filename, 'wb'))

        # Measure prediction time
        start = time()
        prediction_test = model.predict(X_test)
        prediction_train = model.predict(X_train)
        end = time()

        results['Prediction Time'] = end - start

        # Calculate metrics
        results['Accuracy : Test'] = accuracy_score(y_test, prediction_test)
        results['Accuracy : Train'] = accuracy_score(y_train, prediction_train)

        results['F1_Score : Test'] = f1_score(y_test, prediction_test, average='weighted')
        results['F1_Score : Train'] = f1_score(y_train, prediction_train, average='weighted')

        results['Precision : Test'] = precision_score(y_test, prediction_test, average='weighted')
        results['Precision : Train'] = precision_score(y_train, prediction_train, average='weighted')

        results['Recall : Test'] = recall_score(y_test, prediction_test, average='weighted')
        results['Recall : Train'] = recall_score(y_train, prediction_train, average='weighted')

        #I maintained "average = 'weighted'" for the purpose of ease of transitions in case of doing a multiclass classification, for instance in the future maybe trying to create a classification that involves subtle jabs or stuff like that and easier scalability basically

        print(f'Training {model.__class__.__name__} finished in {results["Training Time"]:.2f} sec')

        final_result.append(results.copy())

    return final_result




# %%
final_result = pipeline(training_data, y_train, testing_data, y_test)

# %%
def performfinalresult(final_results):
    results = pd.DataFrame(final_results)
    
    # Reindex the columns
    results = results.reindex(columns=[
        'Algorithm', 'Accuracy : Test', 'Precision : Test', 'Recall : Test', 'F1_Score : Test', 'Prediction Time',
        'Accuracy : Train', 'Precision : Train', 'Recall : Train', 'F1_Score : Train', 'Training Time'
    ])
    
    # Sort the values based on 'F1 Score : Test' in descending order
    results.sort_values(by='F1_Score : Test', inplace=True, ascending=False)

    return results


# %%
results = performfinalresult(final_result)

# %%
results.head(10)
results.reset_index(drop = True)

# %%
results.describe().loc[['min', 'max']]

# %% [markdown]
# Summary in Graph

# %%
# Finding the best scores
best_acc = results[results['Accuracy : Test'] == results['Accuracy : Test'].max()]
best_f1 = results[results['F1_Score : Test'] == results['F1_Score : Test'].max()]
best_precision = results[results['Precision : Test'] == results['Precision : Test'].max()]
best_recall = results[results['Recall : Test'] == results['Recall : Test'].max()]

# Setting the style and figure size
sns.set_style('darkgrid')
plt.figure(figsize=(15, 6))

# Set bar width
barWidth = 0.17

# Set height of bar
bars1 = results['Accuracy : Test']
bars2 = results['F1_Score : Test']

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Make the bar plot
pal = sns.color_palette()
plt.bar(r1, bars1, color=pal[0], width=barWidth, edgecolor='white', label='Test Accuracy')
plt.bar(r2, bars2, color=pal[1], width=barWidth, edgecolor='white', label='Test F1 Score')

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold', fontsize=13)
plt.ylabel('Score', fontweight='bold', fontsize=13)
plt.xticks([r + barWidth for r in range(len(bars1))], results['Algorithm'], rotation=15, fontsize=11)

# Create legend & Show graphic
plt.legend(fontsize=13)

# Adding text box with best scores
textstr = '\n'.join([
    'Best Accuracy: {:.3f} - {}'.format(best_acc['Accuracy : Test'].values[0], best_acc['Algorithm'].values[0]),
    'Best F1 Score: {:.3f} - {}'.format(best_f1['F1_Score : Test'].values[0], best_f1['Algorithm'].values[0])
])
props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

plt.title('Classification Summary of Algorithms', fontweight='bold', fontsize=17)


# %% [markdown]
# Training and prediction time of Algorithm

# %%
best_train_time = results[results['Training Time'] == results['Training Time'].min()]
worst_train_time = results[results['Training Time'] == results['Training Time'].max()]
best_prediction_time = results[results['Prediction Time'] == results['Prediction Time'].min()]
worst_prediction_time = results[results['Prediction Time'] == results['Prediction Time'].max()]

plt.figure(figsize = (12, 7))

barWidth = 0.17
 
# set height of bar
bars1 = results['Training Time']
bars2 = results['Prediction Time']
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color= pal[0], width=barWidth, edgecolor='white', label='Training Time')
plt.bar(r2, bars2, color= pal[1], width=barWidth, edgecolor='white', label='Prediction Time')
 
# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold', fontsize = 13)
plt.ylabel('Time (seconds)', fontweight = 'bold', fontsize = 13)
plt.xticks([r + barWidth for r in range(len(bars1))], results['Algorithm'], rotation = 15, fontsize = 11)


textstr = '\n'.join(('Best Training Time: {:.3f} - {}'.format(best_train_time['Training Time'].values[0], best_train_time['Algorithm'].values[0]), 
                     'Worst Training Time: {:.3f} - {}'.format(worst_train_time['Training Time'].values[0], worst_train_time['Algorithm'].values[0]),
                   'Best Prediction Time: {:.3f} - {}'.format(best_prediction_time['Training Time'].values[0], best_prediction_time['Algorithm'].values[0]), 
                    'Worst Prediction Time: {:.3f} - {}'.format(worst_prediction_time['Training Time'].values[0], worst_prediction_time['Algorithm'].values[0])))
props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

#place a text box
plt.text(3.2, 12, textstr, fontsize=14,  bbox=props) 

# Create legend & Show graphic
plt.legend(fontsize = 13)
plt.title('Training and Prediction time of Algorithms', fontweight = 'bold', fontsize = 17)

# %% [markdown]
# Let's now create the prediction system

# %%
df.head()

# %% [markdown]
# - Load the entire TfidfVectorizer object

# %%
with open('tfidfvectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)


# %%
# Example data to predict
data = ['You are someone I would like to have be with']

# Transform the data using the loaded vectorizer
preprocessed_data = tfidf_vectorizer.transform(data)

# Load the trained model
trained_model = pickle.load(open('LinearSVC.pkl', 'rb'))

# Make predictions
prediction = trained_model.predict(preprocessed_data)

# Print the prediction
if prediction == 1:
    print('bullying')
else:
    print('non-bullying')

# %% [markdown]
# Attempting some fine-tuning

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def tuning(clf, param_dict, X_train, y_train, X_test, y_test):
    scorer = make_scorer(f1_score)

    grid_obj = GridSearchCV(estimator=clf, param_grid=param_dict, scoring=scorer, cv= 5 )

    grid_fit = grid_obj.fit(X_train, y_train)

    best_clf = grid_fit.best_estimator_

    prediction = (clf.fit(X_train, y_train)).predict(X_test)

    best_prediction = best_clf.predict(X_test)

    print(clf.__class__.__name__)
    print(f'Best Parameter: {grid_fit.best_params_}')

    print(f'Accuracy: {accuracy_score(y_test,best_prediction)}')

    print(f'F1 Score: {f1_score(y_test,best_prediction)}')

    print(f'Precision: {precision_score(y_test,best_prediction)}')
    
    print(f'Recall: {recall_score(y_test,best_prediction)}')

# %%
param_grid = {
    'C' : [0.25, 0.5, 0.75, 1, 1.2]
}

clf_model = LinearSVC()

tuning(clf_model, param_grid, training_data, y_train, testing_data, y_test)

# %%
pickle.dump(clf_model, open('LinearSVCTuned.pkl', 'wb'))

# %%



