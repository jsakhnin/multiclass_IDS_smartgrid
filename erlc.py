from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np

'''
Ensemble Representation Learning Classifier (ERLC)
'''

class ERLC:

    def __init__(self, verbose = True, sae_hidden_nodes = 400, outerNN_layers = 2, outerNN_nodes = 1024):
        self.verbose = verbose
        self.sae_hidden_nodes = sae_hidden_nodes
        self.outerNN_layers = outerNN_layers
        self.outerNN_nodes = outerNN_nodes
        self.DT_org = DecisionTreeClassifier()
        self.DT_new = DecisionTreeClassifier()
        self.RF_org = RandomForestClassifier()
        self.RF_new = RandomForestClassifier()
        self.sae = Sequential()
        self.inner_dnn = Sequential()
        self.inner_dnn_new = Sequential()
        self.outer_dnn = Sequential()



    def fit(self, X_train, y_train,  num_classes = 42, sae_epochs = 100, innerNN_epochs = 100, outerNN_epochs = 500):
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
        pca = PCA(n_components = 14)
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
        self.inner_dnn = self.buildInnerDNN(X_train, y_train, num_classes = num_classes, epochs = innerNN_epochs)
        train_DNN = self.inner_dnn.predict_classes(X_train)

        # Build and train inner DNN on new representation
        if (self.verbose):
            print("Training inner DNN on new representation")
        self.inner_dnn_new = self.buildInnerDNN(X_train_new, y_train, num_classes = num_classes, epochs = innerNN_epochs)
        train_DNN_new = self.inner_dnn_new.predict_classes(X_train_new)

        # Changing output of each classifier to categorical
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
        self.outer_dnn = self.buildOuterNN(fused_train, y_train,
                                            num_layers = self.outerNN_layers,
                                            num_nodes = self.outerNN_nodes,
                                            num_classes = num_classes,
                                            do = 0.3,
                                            regularizer = True,
                                            epochs = outerNN_epochs)




    def buildSAE(self, X_train, num_nodes = 400, epochs= 100):
        '''
        This function builds the Stacked AutoEncoder (SAE) and trains it to gain a new representation.
        INPUTS:
        - data: matrix of the data (meter measurements of a smart grid)
        - num_nodes: the number of nodes in the hidden layer (default = 400)
        '''

        input_X= Input(shape=(X_train.shape[1],))
        encoded = Dense(units=800, activation='relu')(input_X)
        encoded = Dense(units=num_nodes, activation='relu')(encoded)
        decoded = Dense(units=800, activation='relu')(encoded)
        decoded = Dense(units=X_train.shape[1], activation='relu')(decoded)
        autoencoder=Model(input_X, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True, validation_split=0.2)

        # Preparing the autoencoder model for use
        model=Sequential()
        model.add(autoencoder.layers[0])
        model.add(autoencoder.layers[1])
        model.add(autoencoder.layers[2])

        return model




    def buildInnerDNN(self, X_train, y_train, num_classes = 42, epochs = 100):
        '''
        This function builds the inner Deep Neural Network (DNN) and trains it to gain a new representation.
        INPUTS:
        - X_train: matrix of the data (meter measurements of a smart grid)
        - y_train: array of the labels for the corresponding X_train samples
        - the autoencoder model built with buildSAE function
        - num_classes: the number of classes
        '''
        # Building the Neural Network
        y_train2 = to_categorical(y_train,num_classes= num_classes)
        nn_model=Sequential()
        nn_model.add(Dense(units=800, activation='relu'))
        nn_model.add(Dense(units=400, activation='relu'))
        nn_model.add(Dense(units=num_classes, activation='softmax'))

        # Early Stop Callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, mode = 'min',patience=5)

        nn_model.compile(optimizer='adam', loss= 'categorical_crossentropy')
        nn_model.fit(X_train, y_train2, epochs = epochs, batch_size=32, validation_split=0.2, callbacks = [earlystop_callback])
        return nn_model



    def buildOuterNN(self, X_train, y_train, num_layers = 2, num_nodes = 1024, num_classes = 42, do = 0.3, regularizer = True, epochs = 500):
        '''
        This function builds the inner Deep Neural Network (DNN) and trains it to gain a new representation.
        INPUTS:
        - X_train: matrix of the data (meter measurements of a smart grid)
        - y_train: array of the labels for the corresponding X_train samples
        - the autoencoder model built with buildSAE function
        - num_classes: the number of classes
        '''
        # Building the Neural Network
        # y_train2 = to_categorical(y_train,num_classes= num_classes)
        outer_nn_model = Sequential()

        for i in range(num_layers):
            if ( (i> 0) & (i < num_layers-1) & (do > 0.0)):
                outer_nn_model.add(Dropout(do))

            if (regularizer == True):
                outer_nn_model.add(Dense(num_nodes, activation='relu', kernel_regularizer= tf.keras.regularizers.l2(0.001)))
            else:
                outer_nn_model.add(Dense(num_nodes, activation='relu'))

        outer_nn_model.add(Dense(num_classes, activation='softmax'))
        outer_nn_model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy')

        # Early Stop Callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, mode = 'min',patience=5)

        outer_nn_model.fit(X_train, y_train, epochs = epochs, batch_size=32, validation_split=0.2, callbacks = [earlystop_callback])
        return outer_nn_model
