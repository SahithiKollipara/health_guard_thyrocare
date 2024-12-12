from tkinter import messagebox, filedialog, simpledialog, Tk, Text, Label, Button, Scrollbar, END
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

main = Tk()
main.title("Prediction and Providing Medication for Thyroid Disease Using Machine Learning Technique (SVM)")
main.geometry("1300x1200")

global classifier
global dataset
global X, Y
global propose_acc, extension_acc
global pca
global X_columns  # Add a global variable to store column names

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset.head))
    preprocessDataset()  # Call preprocessDataset() after uploading the dataset
    trainSVM()  # Call trainSVM() after preprocessing the dataset
    trainOptimizeSVM()  # Call trainOptimizeSVM() after training the SVM

def preprocessBoolean(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == bool:
            dataset[column] = dataset[column].astype(int)
    return dataset

def preprocessDataset():
    global dataset
    global X, Y, X_columns  # Add X_columns here
    text.delete('1.0', END)
    
    # Convert dataset to DataFrame if it's not already
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)
    
    dataset = preprocessBoolean(dataset)  # Preprocess boolean values
    
    # Handle categorical features using one-hot encoding
    dataset = pd.get_dummies(dataset)
    
    dataset_values = dataset.values  # Store the values after preprocessing
    cols = dataset_values.shape[1] - 1
    X = dataset_values[:, 0:cols]
    Y = dataset_values[:, cols]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_columns = dataset.columns.tolist()  # Store column names
    
    text.insert(END, "\nTotal Records after preprocessing are : " + str(len(X)) + "\n")

def trainSVM():
    text.delete('1.0', END)
    global classifier
    global propose_acc
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Number of dataset features (columns) before optimization : " + str(X.shape[1]) + "\n")
    text.insert(END, "Number of records used to train SVM is : " + str(len(X_train)) + "\n")
    text.insert(END, "Number of records used to test SVM is : " + str(len(X_test)) + "\n")
    cls = svm.SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    classifier = cls
    propose_acc = accuracy_score(y_test, prediction_data) * 100
    text.insert(END, "SVM Prediction Accuracy : " + str(propose_acc) + "\n")
    cm = confusion_matrix(y_test, prediction_data)
    text.insert(END, "\nSVM Confusion Matrix\n")
    text.insert(END, str(cm) + "\n")
    fig, ax = plt.subplots()
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_ylim([0, 2])
    plt.show()

def trainOptimizeSVM():
    text.delete('1.0', END)
    global extension_acc
    global X, Y
    global classifier
    global pca
    pca = PCA(n_components=18)
    pca_X = pca.fit_transform(X)
    text.insert(END, "Number of dataset features (columns) after PCA optimization : " + str(pca_X.shape[1]) + "\n")
    X_train, X_test, y_train, y_test = train_test_split(pca_X, Y, test_size=0.2)
    text.insert(END, "Number of records used to train SVM is : " + str(len(X_train)) + "\n")
    text.insert(END, "Number of records used to test SVM is : " + str(len(X_test)) + "\n")
    cls = svm.SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    extension_acc = accuracy_score(y_test, prediction_data) * 100
    classifier = cls
    text.insert(END, "SVM Extension Prediction Accuracy : " + str(extension_acc) + "\n")
    cm = confusion_matrix(y_test, prediction_data)
    text.insert(END, "\nSVM Extension Confusion Matrix\n")
    text.insert(END, str(cm) + "\n")
    fig, ax = plt.subplots()
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_ylim([0, 2])
    plt.show()
    comparisonGraph()  # Call comparisonGraph() after training and optimizing the SVM

def suggestion():
    text1.delete('1.0', END)
    text1.insert(END, "Foods to Avoid\n")
    text1.insert(END, "soy foods: tofu, tempeh, edamame, etc.\n")
    text1.insert(END, "certain vegetables: cabbage, broccoli, kale, cauliflower, spinach, etc.\n")
    text1.insert(END, "fruits and starchy plants: sweet potatoes, cassava, peaches, strawberries, etc.\n")
    text1.insert(END, "nuts and seeds: millet, pine nuts, peanuts, etc.\n\n")
    text1.insert(END, "Foods to Eat\n")
    text1.insert(END, "eggs: whole eggs are best, as much of their iodine and selenium are found in the yolk, while the whites are full of protein\n")
    text1.insert(END, "meat: all meats, including lamb, beef, chicken, etc.\n")
    text1.insert(END, "fish: all seafood, including salmon, tuna, halibut, shrimp, etc.\n")
    text1.insert(END, "vegetables: all vegetables — cruciferous vegetables are fine to eat in moderate amounts, especially when cooked\n")
    text1.insert(END, "fruits: all other fruits, including berries, bananas, oranges, tomatoes, etc.\n\n")

    text1.insert(END, "Medication\n\n")

    text1.insert(END, "The most common treatment is levothyroxine\n")
    text1.insert(END, "(Levoxyl, Synthroid, Tirosint, Unithroid, Unithroid Direct),\n")
    text1.insert(END, "a man-made version of the thyroid hormone thyroxine (T4).\n")
    text1.insert(END, "It acts just like the hormone your thyroid gland normally makes.\n")
    text1.insert(END, "The right dose can make you feel a lot better.")

def comparisonGraph():
    bars = ('Propose SVM Accuracy', 'Extension SVM with PCA Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [propose_acc, extension_acc])
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    global test
    global X1, Y1
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="test")
    test = pd.read_csv(file)
    
    # Preprocess the test dataset
    test = preprocessBoolean(test)
    test = pd.get_dummies(test)
    
    # Fit a new PCA model with the test data
    pca_test = PCA(n_components=18)  # Use the same number of components as the original PCA model
    test_transformed = pca_test.fit_transform(test)
    
    # Predict disease
    y_pred = classifier.predict(test_transformed)
    
    # Display predictions and suggestions
    for i in range(len(test_transformed)):
        if str(y_pred[i]) == '0.0':
            text.insert(END, "X=%s, Predicted = %s" % (test_transformed[i], 'No Thyroid Disease Detected') + "\n\n")
        else:
            text.insert(END, "X=%s, Predicted = %s" % (test_transformed[i], 'Thyroid Disease Risk detected') + "\n\n")
            suggestion()

font = ('times', 14, 'bold')
title = Label(main, text='Prediction and Providing Medication for Thyroid Disease Using Machine Learning Technique (SVM)')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Thyroid Dataset", command=uploadDataset)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

predictButton = Button(main, text="Upload Test Data & Predict Disease", command=predict)
predictButton.place(x=50, y=150)
predictButton.config(font=font1)

text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

text1=Text(main,height=30,width=70)
scroll1=Scrollbar(text1)
text1.configure(yscrollcommand=scroll1.set)
text1.place(x=800,y=200)
text1.config(font=font1)

main.config(bg='mint cream')
main.mainloop()
