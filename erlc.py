from sklearn.base import BaseEstimator
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import backend as K
from sklearn.feature_selection import chi2
from joblib import dump, load
from numpy import savetxt
from numpy import loadtxt

class ERLC (BaseEstimator):

    '''
    Ensemble Representation Learning Classifier (ERLC)
    '''


    def __init__(self, verbose = True, sae_hidden_nodes = 400, innerNN_architecture = [512,512,512],
                    outerNN_architecture = [256,256] ,pca_components = 14):
        self.verbose = verbose
        ## Tunable Parameters
        self.sae_hidden_nodes = sae_hidden_nodes
        self.innerNN_architecture = innerNN_architecture
        self.outerNN_architecture = outerNN_architecture
        self.pca_components = pca_components

        ## Models
        self.DT_org = DecisionTreeClassifier()
        self.DT_new = DecisionTreeClassifier()
        self.RF_org = RandomForestClassifier()
        self.RF_new = RandomForestClassifier()
        self.sae = Sequential()
        self.inner_dnn = Sequential()
        self.inner_dnn_new = Sequential()
        self.outer_dnn = Sequential()

        # Private class variables
        self.isTrained = False
        self.X_train = []
        self.X_train_new = []
        self.y_train = []
        self.fused_train = []
        self.num_classes = 0



    def get_params(self, deep=True):
        return {"sae_hidden_nodes": self.sae_hidden_nodes,
                "pca_components": self.pca_components,
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




    def fit(self, X_train, y_train, sae_epochs = 500, innerNN_epochs = 500, outerNN_epochs = 500):
        '''
        This function fits/trains the model to the inputted data.

        inputs
        --------
        X_train: The training data
        y_train: corresponding labels
        sae_epochs: epochs of training for the Stacked Autoencoder (SAE)
        innerNN_epochs: epochs of training for the inner neural network
        outerNN_epochs: epochs of training for the outer neural network
        '''

        self.X_train = X_train
        self.y_train = y_train
        num_classes = np.max(y_train) + 1
        self.num_classes = num_classes

        if (self.verbose):
            print("Building ERLC model")

        # First we build the autoencoder
        if (self.verbose):
            print("Building autoencoder")
        self.sae = self.buildSAE(X_train, num_nodes = self.sae_hidden_nodes, epochs = sae_epochs)

        # Get new representation
        if (self.verbose):
            print("Getting new representation of the data")
        X_train_new = self.sae.predict(X_train)
        self.X_train_new = X_train_new

        # Train DT on original representation
        if (self.verbose):
            print("Training DT on original representation")
        self.DT_org.fit(X_train, y_train)
        train_DT_org = self.DT_org.predict(X_train)

        # Train DT on new representation
        if (self.verbose):
            print("Training DT on new representation")

        pca = PCA(n_components = self.pca_components)
        Xtr = pca.fit_transform(X_train_new)
        self.DT_new.fit(Xtr,y_train)
        train_DT_new = self.DT_new.predict(Xtr)

        # Train RF on original representation
        if (self.verbose):
            print("Training RF on original representation")
        self.RF_org.fit(X_train, y_train)
        train_RF_org = self.RF_org.predict(X_train)

        # Train RF on new representation
        if (self.verbose):
            print("Training RF on new representation")
        self.RF_new.fit(X_train_new, y_train)
        train_RF_new = self.RF_new.predict(X_train_new)

        # Build and train inner DNN
        if (self.verbose):
            print("Training inner DNN")
        self.inner_dnn = self.buildNN(self.innerNN_architecture, X_train, y_train,
                                            num_classes = num_classes,
                                            activation = 'relu',
                                            do = 0,
                                            epochs = innerNN_epochs)
        train_DNN = self.inner_dnn.predict_classes(X_train)

        # Build and train inner DNN on new representation
        if (self.verbose):
            print("Training inner DNN on new representation")
        self.inner_dnn_new = self.buildNN(self.innerNN_architecture, X_train_new, y_train,
                                            num_classes = num_classes,
                                            activation = 'relu',
                                            do = 0,
                                            epochs = innerNN_epochs)
        train_DNN_new = self.inner_dnn_new.predict_classes(X_train_new)

        # Changing output of each classifier to categorical
        if (self.verbose):
            print("Creating fusion vector")
        train_DT_org = to_categorical(train_DT_org, num_classes = num_classes)
        train_DT_new = to_categorical(train_DT_new, num_classes = num_classes)
        train_RF_org = to_categorical(train_RF_org, num_classes = num_classes)
        train_RF_new = to_categorical(train_RF_new, num_classes = num_classes)
        train_DNN = to_categorical(train_DNN, num_classes = num_classes)
        train_DNN_new = to_categorical(train_DNN_new, num_classes = num_classes)

        # Combining to make fused training data
        fused_train = (train_DT_org, train_DT_new, train_RF_org, train_RF_new, train_DNN, train_DNN_new)
        fused_train = np.concatenate(fused_train, axis=1)
        self.fused_train = fused_train

        # Training outer DNN
        if (self.verbose):
            print("Training outer DNN")
        self.outer_dnn = self.buildNN(self.outerNN_architecture,fused_train, y_train,
                                            num_classes = num_classes,
                                            do = 0.3,
                                            val_split = 0.2,
                                            regularizer = True,
                                            epochs = outerNN_epochs)


        if (self.verbose):
            print("Training complete")

        self.isTrained = True

    def predict (self, X_test):
        '''
        This function predicts the output of the input test data.
        This function must be called after fit has been called.
        inputs
        -------
        X_test: testing data

        outputs
        -------
        y_pred: the predicted labels of the test data
        '''
        # Get new representation of test data
        X_test_new = self.sae.predict(X_test)

        # DT original
        DT_org_test = self.DT_org.predict(X_test)

        # DT new
        pca = PCA(n_components = self.pca_components)
        pca.fit(self.X_train_new)
        tempX = pca.transform(X_test_new)
        DT_new_test = self.DT_new.predict(tempX)

        # RF original
        RF_org_test = self.RF_org.predict(X_test)

        # RF new
        RF_new_test = self.RF_new.predict(X_test_new)

        # DNN original
        DNN_org_test = self.inner_dnn.predict_classes(X_test)

        # DNN new
        DNN_new_test = self.inner_dnn_new.predict_classes(X_test_new)

        # Transform to categorical and combine
        DT_org_test = to_categorical(DT_org_test, num_classes= self.num_classes)
        DT_new_test = to_categorical(DT_new_test, num_classes= self.num_classes)
        RF_org_test = to_categorical(RF_org_test, num_classes= self.num_classes)
        RF_new_test = to_categorical(RF_new_test, num_classes= self.num_classes)
        DNN_org_test = to_categorical(DNN_org_test, num_classes= self.num_classes)
        DNN_new_test = to_categorical(DNN_new_test, num_classes= self.num_classes)

        testSet = (DT_org_test, DT_new_test, RF_org_test, RF_new_test, DNN_org_test, DNN_new_test)
        testSet = np.concatenate(testSet, axis = 1)

        # Outer NN
        y_pred = self.outer_dnn.predict_classes(testSet)

        return y_pred


    def localize (self, X_sample, y_sample, n_measurements = 10, normal_label = 41):
        '''
        This function localizes the attack by returning the score of each feature (measurement) based on its correlation
        with the output of that attack. It uses the chi test function.

        inputs
        -------
        X_sample: the sample vector
        y_sample: the corresponding label
        n_measurements: the top n infected measurements to return
        normal_label: the label value for normal samples

        outputs
        --------
        score: The chi score of each feature
        topIndices: the top n features infected based on the chi score test
        '''

        if (X_sample.ndim > 1):
            raise ValueError('Sample array must be 1 dimensional')
        if (y_sample.ndim > 1):
            raise ValueError('Sample label must be 1 dimensional')
        if (self.isTrained == False):
            raise ValueError('The model has not been trained yet. You must call the fit function first or load a saved model')

        y_pred = self.predict(X_sample)
        chi_score, topF = chi_test(self.X_train, self.y_train, n_measurements = n_measurements)
        row = chi_score[self.y_train==y_pred]
        # currentX = np.vstack( (self.X_train[ (self.y_train == normal_label) | (self.y_train == y_sample) ], X_sample) )
        # currentY = np.hstack( (self.y_train[ (self.y_train == normal_label) | (self.y_train == y_sample) ], y_sample) )

        # score = chi2(currentX, currentY)
        # score = np.nan_to_num(score)
        # score = ch

        # row = score[1,:].copy()
        topIndices = row.argsort()[-n_measurements:][::-1]

        return row, topIndices



    def buildSAE(self, X_train, num_nodes = 400, epochs= 100):
        '''
        This function builds the Stacked AutoEncoder (SAE) and trains it to gain a new representation.

        inputs
        -------
        X_train: matrix of the data
        num_nodes: the number of nodes in the hidden layer
        epochs: number of epochs to train the SAE model

        outputs
        --------
        model: the trained SAE model
        '''

        input_X= Input(shape=(X_train.shape[1],))
        encoded = Dense(units=800, activation='relu')(input_X)
        encoded = Dense(units=num_nodes, activation='relu')(encoded)
        decoded = Dense(units=800, activation='relu')(encoded)
        decoded = Dense(units=X_train.shape[1], activation='relu')(decoded)
        autoencoder=Model(input_X, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        # Early Stop Callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, mode = 'min',patience=10)

        # Fit the autoencoder
        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True, validation_split=0.2, callbacks = [earlystop_callback])

        # Preparing the autoencoder model for use
        model=Sequential()
        model.add(autoencoder.layers[0])
        model.add(autoencoder.layers[1])
        model.add(autoencoder.layers[2])

        return model


    def buildNN(self, architecture, X_train, y_train, num_classes = 42, activation = 'relu', do = 0, regularizer = False, epochs = 500, val_split = 0.2):
        '''
        This function builds the inner Deep Neural Network (DNN) and trains it to gain a new representation.

        inputs
        --------
        X_train: matrix of the data (meter measurements of a smart grid)
        y_train: array of the labels for the corresponding X_train samples
        num_classes: the number of classes
        num_layers: the number of hidden layers in the neural network
        num_nodes: the number of nodes in each hidden layer
        activation: the activation function in each layer (except the final layer)
        do: percent of dropout in between the hidden layers. This should be a value between 0 and 1. If 0, dropout will not be used
        regularizer: whether or not to use l2 regularization in hidden layers
        epochs: number of epochs to train the network
        val_split: percentage of data to use for validation as the network is being trained. This is a value between 0 and 1.

        outputs
        --------
        nn_model: The trained neural network
        '''

        # Building the Neural Network
        y_train2 = to_categorical(y_train,num_classes= num_classes)
        nn_model = Sequential()
        nn_model.add(tf.keras.Input(shape=(X_train.shape[1],)),)

        for i in range(len(architecture)):
            if ( (i> 0) & (i < len(architecture)-1) & (do > 0.0)):
                nn_model.add(Dropout(do))

            if (regularizer == True):
                nn_model.add(Dense(architecture[i], activation=activation, kernel_regularizer= tf.keras.regularizers.l2(0.0001)))
            else:
                nn_model.add(Dense(architecture[i], activation=activation))

        nn_model.add(Dense(num_classes, activation='softmax'))
        nn_model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['acc',self.f1_m])

        # Early Stop Callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, mode = 'min',patience=20)

        if (val_split>0):
            nn_model.fit(X_train, y_train2, epochs = epochs, batch_size=256, validation_split=val_split, callbacks = [earlystop_callback])
        else:
            nn_model.fit(X_train, y_train2, epochs = epochs, batch_size=256, callbacks = [earlystop_callback])

        return nn_model



    def chi_test (self, X, y, n_measurements = 10, normal_label = 41):
        '''
        This function calculates the chi square of features compared to the same features in normal samples. The function takes test data
        and labels, combines them with the training data and labels, then performs chi squared test on each feature.

        inputs
        -------
        X: data matrix
        y: data labels
        n_measurements: the top n infected measurements to return

        outputs
        --------
        final_chi: A matrix of size (labels, features) in which each row corresponds to the chi score of each feature for that attack. The
        labels and features are in the same order as the input data and labels.
        topF: the top n features infected based on the chi score test
        '''

        # Combine saved train data with test data
        # X = np.vstack((self.X_train, X_test))
        # y = np.hstack((self.y_train, y_test))

        labels = np.unique(y)
        numFeatures = X.shape[1]
        final_chi = np.empty( (len(labels)-1, numFeatures) )
        i=0
        normalX = X[y==normal_label]

        for label in labels:
            if (label != normal_label):
                currentX = np.vstack(( X[y==label], normalX) )
                currentY = np.hstack( (y[y==label], y[y==normal_label]) )
                ch, pval = chi2(currentX, currentY)
                final_chi[i,:] = pval
                i=i+1

        final_chi = np.nan_to_num(final_chi)


        topF = []

        for rowNumber in range(np.unique(y).shape[0]-1):
            row = final_chi[rowNumber,:].copy()
            idx = np.argpartition(row, n_measurements)
            topIndices = idx[:n_measurements]
            topF.append(topIndices)

        topF = np.asarray(topF)

        return final_chi, topF

    ## SAVING AND LOADING MODEL
    def save_model(self, save_path = 'saved_model/'):

        # Saving autoencoder
        self.sae.save(save_path+'sae.h5')

        # Saving classifiers
        dump(self.DT_org, save_path + 'DT_org.joblib')
        dump(self.DT_new, save_path + 'DT_new.joblib')
        dump(self.RF_org, save_path + 'RF_org.joblib')
        dump(self.RF_new, save_path + 'RF_new.joblib')

        # Saving neural nets
        self.inner_dnn.save(save_path+'inner_dnn.h5')
        self.inner_dnn_new.save(save_path+'inner_dnn_new.h5')
        self.outer_dnn.save(save_path+'outer_dnn.h5')

        # Saving processed training data
        savetxt(save_path+'X_train.csv', self.X_train, delimiter = ',')
        savetxt(save_path+'X_train_new.csv', self.X_train_new, delimiter = ',')
        savetxt(save_path+'y_train.csv', self.y_train, delimiter = ',')
        savetxt(save_path+'fused_train.csv', self.fused_train, delimiter = ',')

    def load_model(self, save_path = 'saved_model/'):

        # Loading training data
        self.X_train = loadtxt(save_path+'X_train.csv', delimiter = ',')
        self.X_train_new = loadtxt(save_path+'X_train_new.csv', delimiter = ',')
        self.y_train = loadtxt(save_path+'y_train.csv', delimiter = ',')
        self.fused_train = loadtxt(save_path+'fused_train.csv', delimiter = ',')

        # Loading Classifiers
        self.DT_org = load(save_path + 'DT_org.joblib')
        self.DT_new = load(save_path + 'DT_new.joblib')
        self.RF_org = load(save_path + 'RF_org.joblib')
        self.RF_new = load(save_path + 'RF_new.joblib')

        # Loading neural nets
        self.sae = self.rebuildSAE(self.X_train, num_nodes = self.sae_hidden_nodes)
        self.sae.load_weights(save_path + 'sae.h5')

        self.inner_dnn = self.rebuildNN(self.X_train, num_layers = self.innerNN_layers, num_nodes = self.innerNN_nodes,num_classes = np.max(self.y_train)+1,activation = 'relu',do = 0)
        self.inner_dnn.load_weights(save_path + 'inner_dnn.h5')
        self.inner_dnn_new = self.rebuildNN(self.X_train_new, num_layers = self.innerNN_layers, num_nodes = self.innerNN_nodes,num_classes = np.max(self.y_train)+1,activation = 'relu',do = 0)
        self.inner_dnn_new.load_weights(save_path + 'inner_dnn_new.h5')
        self.outer_dnn = self.rebuildNN(self.fused_train, num_layers = self.outerNN_layers, num_nodes = self.outerNN_nodes,num_classes = np.max(self.y_train)+1,activation = 'relu',do = 0.2)
        self.outer_dnn.load_weights(save_path + 'outer_dnn.h5')


    ## Rebuilding functions for loading model
    def rebuildSAE(self, X_train, num_nodes = 400):
        input_X= Input(shape=(X_train.shape[1],))
        encoded = Dense(units=800, activation='relu')(input_X)
        encoded = Dense(units=num_nodes, activation='relu')(encoded)
        decoded = Dense(units=800, activation='relu')(encoded)
        decoded = Dense(units=X_train.shape[1], activation='relu')(decoded)
        autoencoder=Model(input_X, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        model=Sequential()
        model.add(autoencoder.layers[0])
        model.add(autoencoder.layers[1])
        model.add(autoencoder.layers[2])

        return model

    def rebuildNN(self, X_train, num_classes = 42, num_layers = 2, num_nodes = 128, activation = 'relu', do = 0, regularizer = False):
        '''
        This function rebuilds the inner Deep Neural Network (DNN) and trains it to gain a new representation.

        inputs
        --------
        X_train: matrix of the data (meter measurements of a smart grid)
        num_classes: the number of classes
        num_layers: the number of hidden layers in the neural network
        num_nodes: the number of nodes in each hidden layer
        activation: the activation function in each layer (except the final layer)
        do: percent of dropout in between the hidden layers. This should be a value between 0 and 1. If 0, dropout will not be used
        regularizer: whether or not to use l2 regularization in hidden layers

        outputs
        --------
        nn_model: The trained neural network
        '''

        # Building the Neural Network
        nn_model = Sequential()
        nn_model.add(tf.keras.Input(shape=(X_train.shape[1],)),)
        for i in range(num_layers):
            if ( (i> 0) & (i < num_layers-1) & (do > 0.0)):
                nn_model.add(Dropout(do))

            if (regularizer == True):
                nn_model.add(Dense(num_nodes, activation=activation, kernel_regularizer= tf.keras.regularizers.l2(0.0001)))
            else:
                nn_model.add(Dense(num_nodes, activation=activation))

        nn_model.add(Dense(num_classes, activation='softmax'))
        nn_model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['acc',self.f1_m])

        return nn_model

    ## METRICS
    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
