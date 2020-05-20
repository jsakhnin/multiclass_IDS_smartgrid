from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import preprocessing

class DataPipeline():

    def loadData(self, path = 'input/', num_sets = 15):
        '''
        This function loads the data from the path variable provided and returns the data as a pandas dataframe
        '''
        print('Loading dataset #1')
        data = pd.DataFrame(arff.loadarff('{}power_multiclass/data1.arff'.format(path))[0])

        for i in range(2,num_sets+1):
            print('Loading dataset #{}'.format(i))
            dataTemp = pd.DataFrame(arff.loadarff('{}power_multiclass/data{}.arff'.format(path,i))[0])
            data = pd.concat([data,dataTemp],axis=0)

        print("Finished Loading. Final Size = {}".format(data.shape))

        return data

    def getFeatureLabels(self, df):
        '''
        This function takes a dataframe and returns the labels of used features
        '''
        df = df.astype(np.float64)
        label = df['marker'].astype(int)
        df = df.drop(['marker'], axis=1)

        df = df.drop(['snort_log1','snort_log2','snort_log3','snort_log4',
                    'control_panel_log1','control_panel_log2','control_panel_log3','control_panel_log4',
                    'relay1_log','relay2_log','relay3_log','relay4_log'], axis=1)

        # df = df.reset_index()

        return df.columns


    def dataProc(self, df):
        '''
        This function takes a dataframe and splits it into data and labels, proccesses them into numpy arrays and
        splits them into training, validation, and testing data and labels.
        '''
        df = df.astype(np.float64)
        label = df['marker'].astype(int)
        df = df.drop(['marker'], axis=1)

        df = df.drop(['snort_log1','snort_log2','snort_log3','snort_log4',
                  'control_panel_log1','control_panel_log2','control_panel_log3','control_panel_log4',
                  'relay1_log','relay2_log','relay3_log','relay4_log'], axis=1)

        # df = df.reset_index()

        df = df.replace(-np.inf, 0)
        df = df.replace(np.inf, 0)


        # Converting to arrays
        X = np.asarray(df)
        y =  np.asarray(label)

        # Scaling data
        scalar = preprocessing.MinMaxScaler()
        X = scalar.fit_transform(X)

        return X,y
