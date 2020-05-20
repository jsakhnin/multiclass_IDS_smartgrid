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




class ERLC (BaseEstimator):

    '''
    Ensemble Representation Learning Classifier (ERLC)
    '''


    def __init__(self, verbose = True, sae_hidden_nodes = 400, innerNN_layers = 3, innerNN_nodes = 512, outerNN_layers = 2, outerNN_nodes = 256 ,pca_components = 14, X_train = [], y_train = []):
        self.verbose = verbose
        ## Tunable Parameters
        self.sae_hidden_nodes = sae_hidden_nodes
        self.innerNN_layers = innerNN_layers
        self.innerNN_nodes = innerNN_nodes
        self.outerNN_layers = outerNN_layers
        self.outerNN_nodes = outerNN_nodes

        ## Models
        self.DT_org = DecisionTreeClassifier()
        self.DT_new = DecisionTreeClassifier()
        self.RF_org = RandomForestClassifier()
        self.RF_new = RandomForestClassifier()
        self.sae = Sequential()
        self.inner_dnn = Sequential()
        self.inner_dnn_new = Sequential()
        self.outer_dnn = Sequential()
        self.pca = PCA(n_components = pca_components)

        # Private class variables
        self.normal_means = []
        self.normal_std = []
        self.normal_min = []
        self.normal_max = []
        self.isTrained = False
        self.X_train = X_train
        self.y_train = y_train



    def get_params(self, deep=True):
        return {"sae_hidden_nodes": self.sae_hidden_nodes,
                "innerNN_layers": self.innerNN_layers,
                "innerNN_nodes": self.innerNN_nodes,
                "outerNN_layers": self.outerNN_layers,
                "outerNN_nodes": self.outerNN_nodes,
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




    def fit(self, X_train, y_train,  num_classes = 42, sae_epochs = 500, innerNN_epochs = 500, outerNN_epochs = 500):
        '''
        This function fits/trains the model to the inputted data.

        inputs
        --------
        X_train: The training data
        y_train: corresponding labels
        num_classes: number of labels in the data
        sae_epochs: epochs of training for the Stacked Autoencoder (SAE)
        innerNN_epochs: epochs of training for the inner neural network
        outerNN_epochs: epochs of training for the outer neural network
        '''

        self.X_train = X_train
        self.y_train = y_train

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

        # Train DT on original representation
        if (self.verbose):
            print("Training DT on original representation")
        self.DT_org.fit(X_train, y_train)
        train_DT_org = self.DT_org.predict(X_train)

        # Train DT on new representation
        if (self.verbose):
            print("Training DT on new representation")
        Xtr = self.pca.fit_transform(X_train_new)
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
        # self.inner_dnn = self.buildInnerDNN(X_train, y_train, num_classes = num_classes, epochs = innerNN_epochs)
        self.inner_dnn = self.buildNN(X_train, y_train,
                                            num_layers = self.innerNN_layers,
                                            num_nodes = self.innerNN_nodes,
                                            num_classes = num_classes,
                                            activation = 'relu',
                                            do = 0,
                                            # regularizer = True,
                                            epochs = innerNN_epochs)
        train_DNN = self.inner_dnn.predict_classes(X_train)

        # Build and train inner DNN on new representation
        if (self.verbose):
            print("Training inner DNN on new representation")
        self.inner_dnn_new = self.buildNN(X_train_new, y_train,
                                            num_layers = self.innerNN_layers,
                                            num_nodes = self.innerNN_nodes,
                                            num_classes = num_classes,
                                            activation = 'relu',
                                            do = 0,
                                            # regularizer = True,
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

        # Training outer DNN
        if (self.verbose):
            print("Training outer DNN")
        self.outer_dnn = self.buildNN(fused_train, y_train,
                                            num_layers = self.outerNN_layers,
                                            num_nodes = self.outerNN_nodes,
                                            num_classes = num_classes,
                                            do = 0.2,
                                            val_split = 0.2,
                                            # regularizer = True,
                                            epochs = outerNN_epochs)

        if (self.verbose):
            print("Finding attack localization metrics")
        getLocalizationStats(X_train, y_train)

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
        tempX = self.pca.transform(X_test_new)
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
        DT_org_test = to_categorical(DT_org_test, num_classes=42)
        DT_new_test = to_categorical(DT_new_test, num_classes=42)
        RF_org_test = to_categorical(RF_org_test, num_classes=42)
        RF_new_test = to_categorical(RF_new_test, num_classes=42)
        DNN_org_test = to_categorical(DNN_org_test, num_classes=42)
        DNN_new_test = to_categorical(DNN_new_test, num_classes=42)

        testSet = (DT_org_test, DT_new_test, RF_org_test, RF_new_test, DNN_org_test, DNN_new_test)
        testSet = np.concatenate(testSet, axis = 1)

        # Outer NN
        y_pred = self.outer_dnn.predict_classes(testSet)

        return y_pred


    def localize (self, X_sample, y_sample):
        '''
        This function localizes the attack by returning the score of each feature (measurement) based on its correlation
        with the output of that attack. It uses the chi test function.
        '''

        if (X_sample.ndim > 1):
            raise ValueError('Sample array must be 1 dimensional')
        if (y_sample.ndim > 1):
            raise ValueError('Sample label must be 1 dimensional')

        score = self.chi_test(X_sample, y_sample)
        # p_diff_final = p_diff_final[::-1].sort()

        return score



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





    def buildNN(self, X_train, y_train, num_classes = 42, num_layers = 2, num_nodes = 128, activation = 'relu', do = 0, regularizer = False, epochs = 500, val_split = 0):
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

        for i in range(num_layers):
            if ( (i> 0) & (i < num_layers-1) & (do > 0.0)):
                nn_model.add(Dropout(do))

            if (regularizer == True):
                nn_model.add(Dense(num_nodes, activation=activation, kernel_regularizer= tf.keras.regularizers.l2(0.0001)))
            else:
                nn_model.add(Dense(num_nodes, activation=activation))

        nn_model.add(Dense(num_classes, activation='softmax'))
        nn_model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['acc',self.f1_m])

        # Early Stop Callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, mode = 'min',patience=10)

        if (val_split>0):
            nn_model.fit(X_train, y_train2, epochs = epochs, batch_size=128, validation_split=val_split, callbacks = [earlystop_callback])
        else:
            nn_model.fit(X_train, y_train2, epochs = epochs, batch_size=128, callbacks = [earlystop_callback])

        return nn_model



    def chi_test (self, X_test, y_test):
        '''
        This function calculates the chi square of features compared to the same features in normal samples. The function takes test data
        and labels, combines them with the training data and labels, then performs chi squared test on each feature.

        inputs
        -------
        X_test: test data
        y_test: test labels

        outputs
        --------
        final_chi: A matrix of size (labels, features) in which each row corresponds to the chi score of each feature for that attack. The
        labels and features are in the same order as the input data and labels.
        '''

        # Combine saved train data with test data
        X = np.vstack((self.X_train, X_test))
        y = np.hstack((self.y_train, y_test))

        labels = np.unique(y)
        numFeatures = X.shape[1]
        final_chi = np.empty( (len(labels), numFeatures) )
        i=0
        normalX = X[y==41]

        for label in labels:
            currentX = np.vstack(( X[y==label], normalX) )
            currentY = np.hstack( (y[y==label], y[y==41]) )
            ans = chi2(currentX, currentY)
            final_chi[i,:] = ans[1]
            i=i+1

        final_chi = np.nan_to_num(final_chi)

        return final_chi

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
