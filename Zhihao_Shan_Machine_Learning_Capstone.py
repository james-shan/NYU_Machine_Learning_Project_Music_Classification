import random
random.seed(16373762)
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


'''Data Preprocessing'''
"""
# Importing raw data to pandas dataframe
data = pd.read_csv("musicData.csv")
print(data.shape) #check data size
data.info() #check variable names
print(data.head(5))

# Clean missing data
data.dropna(subset = ['music_genre'], inplace = True)

print((data['duration_ms']==-1).sum())
print((data['tempo']=="?").sum())

mask = data['duration_ms'] != -1
group_means = data[mask].groupby('music_genre')['duration_ms'].mean()
data.loc[~mask, 'duration_ms'] = data.loc[~mask, 'music_genre'].apply(lambda x: group_means[x])

data['tempo'] = data['tempo'].replace("?", -1)
data['tempo'] = data['tempo'].astype('float64')
mask = data['tempo'] != -1
group_means = data[mask].groupby('music_genre')['tempo'].mean()
data.loc[~mask, 'tempo'] = data.loc[~mask, 'music_genre'].apply(lambda x: group_means[x])

#One-hot encoding of categorical data
for col in data.columns:
    if col in ['mode','key']:
        # one-hot encode the categorical column
        one_hot_encoded = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, one_hot_encoded], axis=1)
        data.drop(col, axis=1, inplace=True)
data.drop(columns = ['key_A','mode_Major'], inplace=True)

# Encode music genre into number
unique_vals = data['music_genre'].unique()
val_to_num = {val: i for i, val in enumerate(unique_vals)}
data['music_genre'] = data['music_genre'].map(val_to_num)

# Normalize continuous data
cols = ['popularity','acousticness','danceability','duration_ms',\
        'energy','instrumentalness','liveness','loudness',\
        'speechiness','tempo','valence']
for col in cols:
    data[col] = (data[col]-data[col].mean())/data[col].std()
print(data.head(10))
data.to_csv("cleaned_data.csv", index = False)

# Show histogram
data.hist(bins=30,figsize = (20,20))
plt.savefig("Histogram_for_variables.png")
plt.show()
"""
#----------------------------------------------------------------------------------------------------------------------#



'''Logistic Regression'''
def logisticRegression(X_train, y_train,X_test,y_test,plot = True):
    # Fit our data to logistic regression model with multi classes
    logreg = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)

    # Compute the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", accuracy)

    # Convert the true and predicted class labels to binary format using LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_binarized = lb.transform(y_test)

    # Compute the overall AUC score and plot
    overall_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')
    print("Overall AUC:", overall_auc)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if plot == True:
        # Plot the ROC curves for each class
        plt.figure(figsize=(12,8))
        lw = 2
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, color in zip(range(len(lb.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (AUC = {1:0.3f})'
                         ''.format(lb.classes_[i], roc_auc[i]))

        # Plot the macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (AUC = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 color='deeppink', linestyle=':', linewidth=lw)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of Logistic Regression')
        plt.legend(loc="lower right")
        plt.savefig("ROC_Curve_Logistic")
        plt.show()

    return roc_auc


'''Adaboost Model'''
def adaboostModel(X_train,y_train,X_test,y_test,plot = True):
    # Train an AdaBoost classifier
    ada = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=10),n_estimators=100, learning_rate=1)
    ada.fit(X_train, y_train)

    # Predict the test labels
    y_pred = ada.predict(X_test)
    y_pred_proba = ada.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Adaboost Model Accuracy:", accuracy)

    # Convert the true and predicted class labels to binary format using LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_binarized = lb.transform(y_test)

    # Compute the overall AUC score
    overall_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')
    print("Overall AUC:", overall_auc)

    # Compute the ROC curve and plot it
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if plot == True:
        # Plot the ROC curves for each class
        plt.figure(figsize=(12, 8))
        lw = 2
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, color in zip(range(len(lb.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (AUC = {1:0.3f})'
                                                               ''.format(lb.classes_[i], roc_auc[i]))

        # Plot the macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (AUC = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 color='deeppink', linestyle=':', linewidth=lw)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of Adaboost')
        plt.legend(loc="lower right")
        plt.savefig("ROC_Curve_DNN")
        plt.show()



'''Neural Network'''
def DNN(X_train,y_train,X_test,y_test,input_dim, plot = False):
    # Define the deep neural network architecture
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    class MultiClassNet(nn.Module):
        def __init__(self,input_dim):
            super(MultiClassNet, self).__init__()
            self.fc1 = nn.Linear(input_dim,64)
            self.fc2 = nn.Linear(64,32)
            self.fc3 = nn.Linear(32,10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Instantiate the model and set the loss function and optimizer
    model = MultiClassNet(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 20
    batch_size = 64
    train_losses = []
    for epoch in range(num_epochs):
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # Store and print loss
        train_losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch+1}, training loss: {loss.item():.4f}")
    plt.plot(range(num_epochs), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title("Training Loss of Model")
    plt.show()

    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        y_pred = torch.max(output, dim=1)[1].numpy()
        y_pred_proba = torch.softmax(output, dim=1)
        y_pred_proba = y_pred_proba.numpy()

    # Compute the accuracy of the model
    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred)
    print("Neural Network Accuracy:", accuracy)

    # Convert the true and predicted class labels to binary format using LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_binarized = lb.transform(y_test)

    # Compute the overall AUC score
    overall_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')
    print("Overall AUC:", overall_auc)

    # Compute the ROC curve and plot it
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if plot == True:
        # Plot the ROC curves for each class
        plt.figure(figsize=(12, 8))
        lw = 2
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, color in zip(range(len(lb.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (AUC = {1:0.3f})'
                                                               ''.format(lb.classes_[i], roc_auc[i]))

        # Plot the macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (AUC = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 color='deeppink', linestyle=':', linewidth=lw)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of Neural Network')
        plt.legend(loc="lower right")
        plt.savefig("ROC_Curve_DNN")
        plt.show()

    return roc_auc


#---------------------------------------------------------------------------------------------------#

''' Main program without dimension reduction '''

# Test train split
data = pd.read_csv("cleaned_data.csv")
train_data, test_data = train_test_split(data, test_size=5000, stratify=data['music_genre'])
#print("Number of rows of each genre in test set:")
#print(test_data['music_genre'].value_counts())
X = data.drop(columns = ['instance_id','artist_name','track_name','obtained_date','music_genre'])
y = data['music_genre']
X_train = train_data.drop(columns = ['instance_id','artist_name','track_name','obtained_date','music_genre'])
y_train = train_data['music_genre']
X_test = test_data.drop(columns = ['instance_id','artist_name','track_name','obtained_date','music_genre'])
y_test = test_data['music_genre']

# Train model to and evaluate
logisticRegression(X_train,y_train,X_test,y_test)
adaboostModel(X_train,y_train,X_test,y_test)
full_roc_auc = DNN(X_train,y_train,X_test,y_test,23, plot = True)

# Get factor importance for each column in X
import_variable_dict = dict()
min_roc_auc = full_roc_auc
for col in X_train.columns:
    X_train_reduced = X_train.drop(columns = col)
    X_test_reduced = X_test.drop(columns = col)
    roc_auc = DNN(X_train_reduced,y_train,X_test_reduced,y_test,22)
    for key,value in roc_auc.items():
        if value < full_roc_auc[key]:
            min_roc_auc[key] = value
            import_variable_dict[key] = col
print(import_variable_dict)
print(min_roc_auc)


#--------------------------------------------------------------------------------------------------#
'''Dimension Reduction and Clustering'''
# Apply PCA
pca = PCA(n_components=8)
principal_components = pca.fit_transform(X)
df_pca = pd.DataFrame(data = principal_components, columns = ['x1','x2','x3','x4','x5','x6','x7','x8'])
X_train_reducted, X_test_reducted, y_train, y_test = train_test_split(df_pca, y, test_size=0.2, random_state=42)
DNN(X_train_reducted,y_train,X_test_reducted,y_test,8)

# t-SNE to 2 dimensions
tsne = TSNE(n_components=2, perplexity=30,random_state=42)
tsne_results = tsne.fit_transform(X)
np.savetxt('X_after_t-SNE_2.csv', tsne_results, delimiter=',',header = ('column_1','column_2'))

# Plot the t-SNE results
data_reducted = pd.read_csv("X_after_t-SNE_2.csv")
tsne_results = data_reducted.to_numpy()
plt.figure(figsize=(30,20))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='tab10')
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("t-SNE plot of 2 dimensions")
plt.colorbar()
plt.savefig("t-SNE_plot")
plt.show()

# k-Means on t-SNE results
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(tsne_results)
wcss = kmeans.inertia_
print("WCSS:", wcss)

# Plot clustering results
plt.figure(figsize=(30, 20))
unique_labels = np.unique(labels)
for i in unique_labels:
    plt.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], label=i)
plt.title(
    f'Clustering results using t-SNE with perplexity=20 and k-means with 10 clusters')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.savefig('K-Means_Plot')
plt.show()

# Neural Network on t-SNE results
#X_train_reducted, X_test_reducted, y_train, y_test = train_test_split(data_reducted, y, test_size=0.2, random_state=42)
#DNN(X_train_reducted,y_train,X_test_reducted,y_test,2)

#data = pd.read_csv("musicData.csv")
#print(data['music_genre'].unique())
