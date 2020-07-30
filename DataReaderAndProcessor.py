import pandas as pd
from sklearn.impute import SimpleImputer


class DataReaderAndProcessor:


    def read_data(self):
        train_dataset=pd.read_csv('dengue_features_train.csv')
        train_dataset_label=pd.read_csv('dengue_labels_train.csv')


        mean_dataset_train=train_dataset.copy()
        drop_dataset_train=train_dataset.copy()
        drop_dataset_train_label=train_dataset_label.copy()
        mean_dataset_train_label=train_dataset_label.copy()

        drop_dataset_train.reset_index(inplace=True)
        drop_dataset_train_label.reset_index(inplace=True)
        drop_dataset_train=drop_dataset_train.merge(drop_dataset_train_label,  left_on='index', right_on='index')
        del drop_dataset_train['index']
        drop_dataset_train.dropna(inplace=True)
        drop_dataset_train.reset_index(inplace=True)
        del drop_dataset_train['index']
        drop_dataset_train_label=drop_dataset_train['total_cases']
        drop_dataset_train=drop_dataset_train.iloc[:,0:24]
        flag=False
        drop_index=0
        for i in range(1,len(drop_dataset_train)):

            if drop_dataset_train.iloc[i,0]=='iq':
                drop_index=i
                break
        drop_dataset_train_sj=drop_dataset_train.copy()
        drop_dataset_train_sj=drop_dataset_train_sj.iloc[0:drop_index,:]

        drop_dataset_train_label_sj=drop_dataset_train_label.copy()
        drop_dataset_train_label_sj=drop_dataset_train_label_sj.iloc[0:drop_index]

        drop_dataset_train_iq=drop_dataset_train.copy()
        drop_dataset_train_iq=drop_dataset_train_iq.iloc[drop_index:,:]

        drop_dataset_train_label_iq=drop_dataset_train_label.copy()
        drop_dataset_train_label_iq=drop_dataset_train_label_iq.iloc[drop_index:]



        mean_dataset_train.fillna(mean_dataset_train.mean())
        mean_dataset_train.reset_index(inplace=True)
        mean_dataset_train_label.reset_index(inplace=True)
        mean_dataset_train=mean_dataset_train.merge(mean_dataset_train_label,  left_on='index', right_on='index')
        del mean_dataset_train['index']
        mean_dataset_train_label=mean_dataset_train['total_cases']
        mean_dataset_train=mean_dataset_train.iloc[:,0:24]

        flag=False
        index_mean=0
        for i in range(1,len(mean_dataset_train)):

            if mean_dataset_train.iloc[i,0]=='iq':
                index_mean=i
                break
        mean_dataset_train_sj=mean_dataset_train.copy()
        mean_dataset_train_sj=mean_dataset_train_sj.iloc[0:index_mean,:]

        mean_dataset_train_label_sj=mean_dataset_train_label.copy()
        mean_dataset_train_label_sj=mean_dataset_train_label_sj.iloc[0:index_mean]

        mean_dataset_train_iq=mean_dataset_train.copy()
        mean_dataset_train_iq=mean_dataset_train_iq.iloc[index_mean:,:]

        mean_dataset_train_label_iq=mean_dataset_train_label.copy()
        mean_dataset_train_label_iq=mean_dataset_train_label_iq.iloc[index_mean:]


        """
        print(mean_dataset_train_iq)
        print(mean_dataset_train_label_iq)
        print(mean_dataset_train_sj)
        print(mean_dataset_train_label_sj)
        print(drop_dataset_train_iq)
        print(drop_dataset_train_label_iq)
        print(drop_dataset_train_sj)
        print(drop_dataset_train_label_sj)
        
        
        """
        print(type(drop_dataset_train_label_sj),type(drop_dataset_train_label_iq),type(mean_dataset_train_label_iq),type(mean_dataset_train_label_iq))
        frame = {'total_cases':drop_dataset_train_label_sj }
        drop_dataset_train_label_sj=pd.DataFrame(frame)
        frame={}
        frame = {'total_cases': drop_dataset_train_label_iq}
        drop_dataset_train_label_iq = pd.DataFrame(frame)
        frame = {}
        frame = {'total_cases': mean_dataset_train_label_sj}
        mean_dataset_train_label_sj = pd.DataFrame(frame)
        frame = {}
        frame = {'total_cases': mean_dataset_train_label_iq}
        mean_dataset_train_label_iq = pd.DataFrame(frame)
        frame = {}
        drop_dataset_train_sj.reset_index(inplace=True)
        del drop_dataset_train_sj['index']
        drop_dataset_train_label_sj.reset_index(inplace=True)
        del drop_dataset_train_label_sj['index']
        drop_dataset_train_iq.reset_index(inplace=True)
        del drop_dataset_train_iq['index']
        drop_dataset_train_label_iq.reset_index(inplace=True)
        del drop_dataset_train_label_iq['index']
        mean_dataset_train_sj.reset_index(inplace=True)
        del mean_dataset_train_sj['index']
        mean_dataset_train_label_sj.reset_index(inplace=True)
        del mean_dataset_train_label_sj['index']
        mean_dataset_train_iq.reset_index(inplace=True)
        del mean_dataset_train_iq['index']
        mean_dataset_train_label_iq.reset_index(inplace=True)
        del mean_dataset_train_label_iq['index']


        return drop_dataset_train,drop_dataset_train_label,mean_dataset_train,mean_dataset_train_label



print(DataReaderAndProcessor.read_data(DataReaderAndProcessor))
