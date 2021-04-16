import os
import time
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import erlc
import dataPipeline as dp
from scipy.stats.stats import pearsonr


def load_and_process():
    dataPipeline = dp.DataPipeline()
    data = dataPipeline.loadData()
    X, y = dataPipeline.dataProc(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def basic_implementation():
    X_train, X_test, y_train, y_test = load_and_process()
    model =  erlc.ERLC()
    try:
        model.load_model(save_path = 'saved_model/')
    except:
        model.fit(X_train, y_train)
        model.save_model()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Accuracy = {}, F1-score = {}, MCC = {}".format(acc, f1, mcc))
    print("TEST COMPLETE \n")


def features_heatmap():
    X_train, X_test, y_train, y_test = load_and_process()
    final_chi, top_features = erlc.ERLC.chi_test(X[:,1:X.shape[1]], y)
    # Create heatmap
    plt.figure(figsize=(24,12))
    x_axis_labels = np.unique(y)
    x_axis_labels = x_axis_labels[:-1]
    y_axis_labels = dataPipeline.getFeatureLabels(data)
    sns.heatmap(final_chi,
                xticklabels = range(1,117),
                yticklabels = x_axis_labels,
                cmap="Blues", 
                cbar_kws={"shrink": 1, "orientation": "horizontal", "pad": 0.1, "fraction": 0.02},
                linewidths = 0.8)
    plt.ylabel("Label", fontsize=20)
    plt.xlabel("Feature", fontsize=20)
    plt.savefig('output/figures/chi_heatmap.pdf')
    print("TEST COMPLETE \n")


def localizationTest(X, y, test_split=0.2):
    '''
    This function performs localization test by splitting the data and finding the correlation of the p-values
    from the chi test. 
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split)
    
    # performing chi test on train and test 
    final_chi_tr, top_features_tr = erlc.ERLC.chi_test(X_train[:,1:X.shape[1]], y_train)
    final_chi_te, top_features_te = erlc.ERLC.chi_test(X_test[:,1:X.shape[1]], y_test)
    
    attack_label = 2
    labels = np.unique(y)
    i = 0

    decimals = 4
    corr = []
    attack = []

    for i in range(final_chi_tr.shape[0]):
        temp = pearsonr(final_chi_tr[i],final_chi_te[i])
        temp = round(temp[0], decimals)
        corr.append(temp)
        attack.append(labels[i])
        
    return attack, corr


def pearson_correlations(test_split=0.2):
    '''
    Test localization using pearson correlation
    '''
    dataPipeline = dp.DataPipeline()
    data = dataPipeline.loadData()
    X, y = dataPipeline.dataProc(data)
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = test_split)

    attack, corr = localizationTest(X, y)
        
    # Process and save results
    corr_result = pd.DataFrame({
        'Attack Label': attack,
        'Pearson Correlation': corr,
    })
    corr_result.to_csv('output/results_pearsonCorr.csv', index=False)
    corr_result.to_csv('output/results_pearsonCorrelation1.txt', sep = '&', index=False)
    # Adding "\\" at the end of each line for latex
    with open('output/results_pearsonCorrelation1.txt', 'r') as istr:
        with open('output/results_pearsonCorrelation.txt', 'w') as ostr:
            for line in istr:
                line = line.rstrip('\n') + '\\\\'
                print(line, file=ostr)
    # Remove old file
    os.remove('output/results_pearsonCorrelation1.txt')
    avg_corr = np.mean(corr)
    max_corr = np.max(corr)
    min_corr = np.min(corr)
    labels = np.unique(y)

    print("Mean : {}".format(avg_corr))
    print("Min: {} for label {}".format(min_corr, labels[np.where(corr==min_corr)] ))
    print("Max: {} for label {}".format(max_corr, labels[np.where(corr==max_corr)] ))
    print("TEST COMPLETE \n")

def pearson_correlations_split():
    '''
    Pearson correlations of chi square localization for varying train-test splits
    '''
    dataPipeline = dp.DataPipeline()
    data = dataPipeline.loadData()
    X, y = dataPipeline.dataProc(data)
    splits = np.arange(0.1,0.55,0.05)
    averages = []

    for split in splits:
        attack, corr = localizationTest(X, y, test_split = split)
        avg_corr = np.mean(corr)
        averages.append(avg_corr)

    avg_result = pd.DataFrame({
        'Split': splits,
        'Average Correlation': averages,
    })
    avg_result.to_csv('output/results_avgCorrelationPerSplit.csv', index=False)


## -------------------- Cross validation
def runCV(model, X, y, numFolds=10):
    '''
    Run cross validation on given model
    '''
    cv = KFold(numFolds, True, 1)
    acc = []
    f1 = []
    mcc = []
    trainT = []
    testT = []
    
    name = model.__class__.__name__
    print("Performing CV on {}".format(name))
    i = 1;
    for train, test in cv.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        print("Train shape {}".format(X_train.shape))
        print("Test shape {}".format(X_test.shape))

        time1 = time.time()
        model.fit(X_train, y_train)
        time2 = time.time()
        trainTime= time2-time1
        
        time1 = time.time()
        y_pred = model.predict(X_test)
        time2 = time.time()
        predictTime = time2-time1
        
        acc_temp = accuracy_score(y_test, y_pred)
        f1_temp = f1_score(y_test, y_pred, average='weighted')
        mcc_temp = matthews_corrcoef(y_test, y_pred)
        
        print("Fold #{}".format(i))
        print("Accuracy = {}, F1-score = {}, MCC = {}, Train Time = {}".format(acc_temp, f1_temp, mcc_temp, trainTime))
        
        
        acc.append(acc_temp)
        f1.append(f1_temp)
        mcc.append(mcc_temp)
        trainT.append(trainTime)
        testT.append(predictTime)
        i=i+1
    
    return acc, f1, mcc, trainT, testT

def cv_experiment(num_folds=10):
    '''
    run cross validation on all models
    '''
    dataPipeline = dp.DataPipeline()
    data = dataPipeline.loadData()
    X, y = dataPipeline.dataProc(data)
    model =  erlc.ERLC()
    try:
        model.load_model(save_path = 'saved_model/')
    except:
        model.fit(X_train, y_train)
        model.save_model()
    models = [
        model,
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(), 
    ]
    writer = pd.ExcelWriter('output/Results_{}CV2.xlsx'.format(num_folds))
    for model in models:
        name = model.__class__.__name__
        acc, f1 , mcc, trainT, testT = runCV(model,X,y, numFolds = num_folds) 
        df = pd.DataFrame({
            'Accuracy': acc,
            'F1-Score': f1,
            'MCC': mcc,
            'Training Time': trainT,
            'Testing Time': testT
        })
        df.to_excel(writer, name)
        
    writer.save()


## ------------------------------- Scalability
def evaluate_feature_percentage (model, X, y, feature_percentage = 1.0):
    # Extract random features from data
    num_features = round(feature_percentage * X.shape[1]).astype(int)
    idx = random.sample(range(X.shape[1]), num_features)
    idx = np.sort(idx)
    X = X[:, idx]
    print("Testing on {} features".format(X.shape[1]))
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Training model
    time1 = time.time()
    model.fit(X_train, y_train)
    time2 = time.time()
    train_time = time2 - time1
    
    # Testing model
    time1 = time.time()
    y_pred = model.predict(X_test)
    time2 = time.time()
    test_time = time2 - time1
    
    # Calculating metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Accuracy = {}, F1-score = {}, MCC = {}".format(acc, f1, mcc))
    return train_time, test_time, acc, f1, mcc

def scalability_analysis (model, X, y, feature_percentages = np.linspace(0.1,1, 10) ):
    train_times, test_times, acc_scores, f1_scores, mcc_scores = [], [], [], [], []
    
    for fp in feature_percentages:
        train_time, test_time, acc, f1, mcc = evaluate_feature_percentage(model, X, y, feature_percentage = fp)
        train_times.append(train_time)
        test_times.append(test_time)
        acc_scores.append(acc)
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        
    return train_times, test_times, acc_scores, f1_scores, mcc_scores

def scalability_test():
    dataPipeline = dp.DataPipeline()
    data = dataPipeline.loadData()
    X, y = dataPipeline.dataProc(data)

    model =  erlc.ERLC()
    try:
        model.load_model(save_path = 'saved_model/')
    except:
        model.fit(X_train, y_train)
        model.save_model()
    models = [
        model,
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(), 
    ]

    fp_range = np.arange(0.1,1.1,0.1)
    writer = pd.ExcelWriter('output/Results_scalabilityAnalysis_{}intervals.xlsx'.format(len(fp_range)))
    for model in MODELS:
        model_name = model.__class__.__name__
        # Perform scalability analysis on current model
        train_times, test_times, acc_scores, f1_scores, mcc_scores = \
        scalability_analysis(model, X, y, feature_percentages = fp_range )
        # Add to excel file
        df = pd.DataFrame({
            'Feature Percentage': fp_range,
            'Accuracy': acc_scores,
            'F1-Score': f1_scores,
            'MCC': mcc_scores,
            'Training Time': train_times,
            'Testing Time': test_times
        })
        df.to_excel(writer, model_name)
           
    writer.save()


## --------------------------------- Confusion matrix
def cmatrix():
    X_train, X_test, y_train, y_test = load_and_process()
    model =  erlc.ERLC()
    try:
        model.load_model(save_path = 'saved_model/')
    except:
        model.fit(X_train, y_train)
        model.save_model()

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Size of test data: {X_test.shape}")
    print(f"Size of test labels: {y_test.shape}")
    print(conf_matrix)
    # Plot    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='small')
    
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('Actual Label', fontsize=16)
    plt.savefig('output/figures/cmatrix.pdf')