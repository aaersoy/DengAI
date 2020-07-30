import warnings

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.impute import SimpleImputer
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


"""Data okuma"""
def read_data():
    train_dataset = pd.read_csv('dengue_features_train.csv')
    train_dataset_label = pd.read_csv('dengue_labels_train.csv')
    test_dataset=pd.read_csv('dengue_features_test.csv')
    del train_dataset['week_start_date']
    del test_dataset['week_start_date']
    #del train_dataset['year']
    #del test_dataset['year']
    test_dataset.fillna(test_dataset.mean(), inplace=True)

    mean_dataset_train = train_dataset.copy()
    drop_dataset_train = train_dataset.copy()
    drop_dataset_train_label = train_dataset_label.copy()
    mean_dataset_train_label = train_dataset_label.copy()



    drop_dataset_train.reset_index(inplace=True)
    drop_dataset_train_label.reset_index(inplace=True)
    drop_dataset_train = drop_dataset_train.merge(drop_dataset_train_label, left_on='index', right_on='index')
    del drop_dataset_train['index']
    drop_dataset_train.dropna(inplace=True)
    drop_dataset_train.reset_index(inplace=True)
    del drop_dataset_train['index']
    drop_dataset_train_label = drop_dataset_train['total_cases']
    drop_dataset_train = drop_dataset_train.iloc[:, 0:27]
    del drop_dataset_train['year_y']
    del drop_dataset_train['weekofyear_y']
    del drop_dataset_train['city_y']
    flag = False
    drop_index = 0
    for i in range(1, len(drop_dataset_train)):

        if drop_dataset_train.iloc[i, 0] == 'iq':
            drop_index = i
            break
    drop_dataset_train_sj = drop_dataset_train.copy()
    drop_dataset_train_sj = drop_dataset_train_sj.iloc[0:drop_index, :]

    drop_dataset_train_label_sj = drop_dataset_train_label.copy()
    drop_dataset_train_label_sj = drop_dataset_train_label_sj.iloc[0:drop_index]

    drop_dataset_train_iq = drop_dataset_train.copy()
    drop_dataset_train_iq = drop_dataset_train_iq.iloc[drop_index:, :]

    drop_dataset_train_label_iq = drop_dataset_train_label.copy()
    drop_dataset_train_label_iq = drop_dataset_train_label_iq.iloc[drop_index:]


    mean_dataset_train.reset_index(inplace=True)
    mean_dataset_train_label.reset_index(inplace=True)
    mean_dataset_train = mean_dataset_train.merge(mean_dataset_train_label, left_on='index', right_on='index')
    del mean_dataset_train['index']
    mean_dataset_train_label = mean_dataset_train['total_cases']
    mean_dataset_train = mean_dataset_train.iloc[:, 0:27]
    del mean_dataset_train['year_y']
    del mean_dataset_train['weekofyear_y']
    del mean_dataset_train['city_y']
    mean_dataset_train.fillna(mean_dataset_train.mean(), inplace=True)



    flag = False
    index_mean = 0
    for i in range(1, len(mean_dataset_train)):

        if mean_dataset_train.iloc[i, 0] == 'iq':
            index_mean = i
            break
    mean_dataset_train_sj = mean_dataset_train.copy()
    mean_dataset_train_sj = mean_dataset_train_sj.iloc[0:index_mean, :]

    mean_dataset_train_label_sj = mean_dataset_train_label.copy()
    mean_dataset_train_label_sj = mean_dataset_train_label_sj.iloc[0:index_mean]

    mean_dataset_train_iq = mean_dataset_train.copy()
    mean_dataset_train_iq = mean_dataset_train_iq.iloc[index_mean:, :]

    mean_dataset_train_label_iq = mean_dataset_train_label.copy()
    mean_dataset_train_label_iq = mean_dataset_train_label_iq.iloc[index_mean:]

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

    frame = {'total_cases': drop_dataset_train_label_sj}
    drop_dataset_train_label_sj = pd.DataFrame(frame)
    frame = {}
    frame = {'total_cases': drop_dataset_train_label_iq}
    drop_dataset_train_label_iq = pd.DataFrame(frame)
    frame = {}
    frame = {'total_cases': mean_dataset_train_label_sj}
    mean_dataset_train_label_sj = pd.DataFrame(frame)
    frame = {}
    frame = {'total_cases': mean_dataset_train_label_iq}
    mean_dataset_train_label_iq = pd.DataFrame(frame)
    frame = {}
    frame = {'total_cases': drop_dataset_train_label}
    drop_dataset_train_label = pd.DataFrame(frame)
    frame = {}
    frame = {'total_cases': mean_dataset_train_label}
    mean_dataset_train_label = pd.DataFrame(frame)
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

    labelencoder1 = LabelEncoder()
    drop_dataset_train['year_x'] = labelencoder1.fit_transform(drop_dataset_train['year_x'])
    drop_dataset_train['city_x'] = labelencoder1.fit_transform(drop_dataset_train['city_x'])
    drop_dataset_train['weekofyear_x'] = labelencoder1.fit_transform(drop_dataset_train['weekofyear_x'])
    mean_dataset_train['year_x'] = labelencoder1.fit_transform(mean_dataset_train['year_x'])
    mean_dataset_train['city_x'] = labelencoder1.fit_transform(mean_dataset_train['city_x'])
    mean_dataset_train['weekofyear_x'] = labelencoder1.fit_transform(mean_dataset_train['weekofyear_x'])
    test_dataset['year'] = labelencoder1.fit_transform(test_dataset['year'])
    test_dataset['city'] = labelencoder1.fit_transform(test_dataset['city'])
    test_dataset['weekofyear'] = labelencoder1.fit_transform(test_dataset['weekofyear'])
    test_dataset['year']+=18  #Train dataseti 2010 a kadar gidiyor ve 20 farklı değer alıyor yıl için. Bu dataset 2008 den başlıyor ve 5 farklı değer alıyor.
    del  drop_dataset_train['total_cases']
    del mean_dataset_train['total_cases']
    return drop_dataset_train, drop_dataset_train_label, mean_dataset_train, mean_dataset_train_label,test_dataset



fold_number=0
drop_dataset_train, drop_dataset_train_label, mean_dataset_train, mean_dataset_train_label,test_data =read_data()


"""foldlara ayırabilmek için bir method, ayırma yapacağı datasetleri, kaçıncı foldu ayıracağını ve fold sayısını alır.Foldları döner."""
def partition_folds(dataset_train,dataset_train_label,i,fold_number):

    for i in range(1,fold_number+1):



        if i*(int(len(dataset_train)/fold_number)) < len(dataset_train)-fold_number:
            temp_test = dataset_train.iloc[(i - 1) * (int(len(dataset_train) / fold_number)):i * (int(len(dataset_train) / fold_number)), :]
        else:
            temp_test = dataset_train.iloc[(i - 1) * (int(len(dataset_train) / fold_number)):, :]

        if i * (int(len(dataset_train_label) / fold_number)) < len(dataset_train_label) - fold_number:
            temp_test_label = dataset_train_label.iloc[(i - 1) * (int(len(dataset_train_label) / fold_number)):i * (
                int(len(dataset_train_label) / fold_number)), :]
        else:
            temp_test_label = dataset_train_label.iloc[(i - 1) * (int(len(dataset_train_label) / fold_number)):, :]


        dataset_train.reset_index(inplace=True)
        temp_test.reset_index(inplace=True)
        temp_train = pd.concat([dataset_train, temp_test, temp_test]).drop_duplicates(keep=False)
        del temp_train['index']
        del temp_test['index']
        temp_train.reset_index(inplace=True)
        del dataset_train['index']
        del temp_train['index']




        dataset_train_label.reset_index(inplace=True)
        temp_test_label.reset_index(inplace=True)
        temp_train_label = pd.concat([dataset_train_label, temp_test_label, temp_test_label]).drop_duplicates(keep=False)
        del dataset_train_label['index']
        del temp_test_label['index']
        del temp_train_label['index']

        temp_test.reset_index(inplace=True)
        temp_train_label.reset_index(inplace=True)
        del temp_test['index']
        del temp_train_label['index']



    return temp_train,temp_train_label,temp_test,temp_test_label

def backward_subset_selection(train_dataset,train_dataset_label,dataset_features,fold_number,model):
    print("backward_subset_selection_linear_regression")

    if model == 'multiple_linear':
        global_mse=model_evaluate_with_cross_validation(train_dataset,train_dataset_label,fold_number,model)
    elif model == 'svm':
        global_mse=model_evaluate_with_cross_validation(train_dataset,train_dataset_label,fold_number,model)
    elif model == 'rf':
        global_mse=model_evaluate_with_cross_validation(train_dataset,train_dataset_label,fold_number,model)

    dataset_features=dataset_features.to_list()
    removed_features=[]

    for k in range(len(dataset_features)):

        t=0
        for i in dataset_features:
            train_temp=train_dataset.copy()
            del train_temp[i]

            mse=model_evaluate_with_cross_validation(train_temp,train_dataset_label,fold_number,model)

            if mse<global_mse:
                t+=1
                global_mse=mse
                max_feature=i


        if t!=0:

            del train_dataset[max_feature]
            dataset_features.remove(max_feature)
            removed_features.append(max_feature)
        elif k==0:
            continue
        else:
            print("break")
            break
    return train_dataset,removed_features

"""Genetik algoritma ile Linear Regression için subset selectionu. Genetik ALgoritmaya göre en iyi olan çözümü döner."""
def subset_selection_with_genetic_algorithm(train_dataset,train_dataset_label,dataset_features,fold_number,cromozome_count,generation_count,model):
    print("subset_selection_with_genetic_algorithm_for_linear_regression")

    dataset_features = dataset_features.to_list()
    elitizm_rate=0.30
    cross_over_rate=0.80
    mutation_rate=0.2
    """İlk kullanılacak populasyonu üretir."""
    def initial_population(cromozome_count,dataset_features):
        cromozomes={}
        for i in range(cromozome_count):
            cromozomes[i]=[]
            for k in range(len(dataset_features)):
                if random.random()<0.5:
                    cromozomes[i].append(0)
                else:
                    cromozomes[i].append(1)
        return cromozomes

    """Verilen parametrelere göre yeni bir nesil döner."""
    def cross_over_and_mutation_section(generation, cross_over_rate, elitizm_dict,mutation_rate,generation_fitness):
        cromozom_index = 0
        next_generation={}
        mean_of_elite=0
        for i in elitizm_dict:
            next_generation[cromozom_index]=[]
            next_generation[cromozom_index] = generation[i]
            mean_of_elite+=generation_fitness[i]
            cromozom_index = cromozom_index + 1
        mean_of_elite=mean_of_elite/len(elitizm_dict)
        while cromozom_index < len(generation):

            first_candidate_ind = []
            second_candidate_ind = []
            first_candidate_ind = generation[int(random.random() * len(generation))]
            second_candidate_ind = generation[int(random.random() * len(generation))]

            if random.random() < cross_over_rate:

                random_gene = int(random.random() * len(first_candidate_ind))
                next_generation[cromozom_index] = first_candidate_ind[0:random_gene] + second_candidate_ind[random_gene:len(second_candidate_ind)]

                """Mutation for individual 1"""
                if random.random() < mutation_rate:

                    if next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] == 0:
                        next_generation[cromozom_index][int(random.random()*len(next_generation[cromozom_index]))] = 1

                    else:
                        next_generation[cromozom_index][int(random.random()*len(next_generation[cromozom_index]))] = 0
                    if next_generation[cromozom_index][
                        int(random.random() * len(next_generation[cromozom_index]))] == 0:
                        next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                    else:
                        next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0
                    if next_generation[cromozom_index][
                        int(random.random() * len(next_generation[cromozom_index]))] == 0:
                        next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                    else:
                        next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0
                    if next_generation[cromozom_index][
                        int(random.random() * len(next_generation[cromozom_index]))] == 0:
                        next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                    else:
                        next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0

                if evaluate_cromozome(next_generation[cromozom_index],mean_of_elite,model):
                    cromozom_index+=1


                """Mutation for individual 2"""
                if cromozom_index < len(next_generation):

                    next_generation[cromozom_index] = second_candidate_ind[0:random_gene] + first_candidate_ind[random_gene:len(first_candidate_ind)]

                    if random.random() < mutation_rate:
                        if next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] == 0:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                        else:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0

                        if next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] == 0:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                        else:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0

                        if next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] == 0:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                        else:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0

                        if next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] == 0:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 1

                        else:
                            next_generation[cromozom_index][int(random.random() * len(next_generation[cromozom_index]))] = 0

                    if evaluate_cromozome(next_generation[cromozom_index], mean_of_elite,model):
                        cromozom_index += 1


        return next_generation


    def evaluate_cromozome(cromosome,mean_of_elite,model):


        temp_train = train_dataset.copy()

        for k in range(len(cromosome)):

            if  cromosome[k]== 0:
                # print(dataset_features[k])
                del temp_train[dataset_features[k]]

                # print(temp_train)
            else:
                continue

        if model == 'multiple_linear':
            success = model_evaluate_with_cross_validation(temp_train, train_dataset_label, fold_number,
                                                                         model)
        elif model == 'svm':
            success= model_evaluate_with_cross_validation(temp_train, train_dataset_label, fold_number,
                                                                         model)
        elif model == 'rf':
            success=model_evaluate_with_cross_validation(temp_train, train_dataset_label, fold_number,
                                                                         model)
        if success<=mean_of_elite:
            return True
        else:
            return False



    """Üretilen generationu değerlendirir ve en iyilerini belirler."""
    def fitness_and_elites(generation,train_dataset,train_dataset_label,dataset_features,elitizm_rate,fold_number,model):
        generation_fitness={}
        for i in generation:

            generation_fitness[i]=[]
            temp_train=train_dataset.copy()

            for k in range(len(generation[i])):

                if generation[i][k]==0:
                    #print(dataset_features[k])
                    del temp_train[dataset_features[k]]

                    #print(temp_train)
                else:
                    continue

            if model=='multiple_linear':
                generation_fitness[i]=model_evaluate_with_cross_validation(temp_train,train_dataset_label,fold_number,model)
            elif model=='svm':
                generation_fitness[i]=model_evaluate_with_cross_validation(temp_train,train_dataset_label,fold_number,model)
            elif model=='rf':
                generation_fitness[i]=model_evaluate_with_cross_validation(temp_train,train_dataset_label,fold_number,model)

        elitizm_dict = []
        print(generation_fitness)
        for z in range(int(len(generation_fitness) * elitizm_rate)):

            min = 100000000000000
            min_index = 0
            for k in range(len(generation_fitness)):
                if k not in elitizm_dict:
                    if generation_fitness[k] < min:
                        min = generation_fitness[k]
                        min_index = k

            elitizm_dict.append(min_index)

        return elitizm_dict,generation_fitness




    generation=initial_population(cromozome_count, dataset_features)
    elitizm_dict,generation_fitness=fitness_and_elites(generation,train_dataset,train_dataset_label,dataset_features,elitizm_rate,fold_number,model)
    for i in range(generation_count):
        generation=cross_over_and_mutation_section(generation,cross_over_rate,elitizm_dict,mutation_rate,generation_fitness)
        elitizm_dict,generation_fitness=fitness_and_elites(generation,train_dataset,train_dataset_label,dataset_features,elitizm_rate,fold_number,model)


    return generation[elitizm_dict[0]]

"""Foldlarla, linear regression değerlendirmesi. Input olarak datasetleri ve kaç tane folda ayıracağını alır. Mse döner."""
def model_evaluate_with_cross_validation(train_dataset,train_dataset_label,fold_number,model):

    mse = 0
    i=1
    for i in range(1,fold_number+1):
        fold_train,fold_train_label,fold_test,fold_test_label=partition_folds(train_dataset,train_dataset_label,i,fold_number)
        if model == 'multiple_linear':
            lr = LinearRegression()
            lr.fit(fold_train, fold_train_label)
            predict_l = lr.predict(fold_test)
            predict = []
            for item in predict_l:
                predict.append(item[0])
        elif model == 'svm':
            svg_reg = SVR(kernel='rbf')
            svg_reg.fit(fold_train, fold_train_label)
            predict = svg_reg.predict(fold_test)
        elif model == 'rf':
            rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
            rf_reg.fit(fold_train,fold_train_label)
            predict = rf_reg.predict(fold_test)

        """X = fold_test.copy()
        print(X)
        print(fold_train_label)
        X.reset_index(inplace=True)
        plt.scatter(X['index'], fold_test_label['total_cases'], color='red')
        plt.plot(X['index'], predict, color='blue')
        plt.show()"""

        difference_array = np.subtract(predict, fold_test_label.to_numpy())
        squared_array = np.square(difference_array)
        mse = mse + squared_array.mean()


    return mse/fold_number

def ensemble_model_evaluate_with_cross_validation(train_dataset_lr,train_dataset_label_lr,train_dataset_svr,train_dataset_label_svr, train_dataset_rf,train_dataset_label_rf,fold_number):

    mse = 0
    mse_2=0
    mse_lr=0
    mse_svr=0
    mse_rf=0
    i = 1
    for i in range(1, fold_number + 1):
        fold_train_lr, fold_train_label_lr, fold_test_lr, fold_test_label_lr = partition_folds(train_dataset_lr, train_dataset_label_lr,
                                                                                   i, fold_number)
        fold_train_svr, fold_train_label_svr, fold_test_svr, fold_test_label_svr = partition_folds(train_dataset_svr,
                                                                                               train_dataset_label_svr,
                                                                                               i, fold_number)
        fold_train_rf, fold_train_label_rf, fold_test_rf, fold_test_label_rf = partition_folds(train_dataset_rf,
                                                                                                   train_dataset_label_rf,
                                                                                                   i, fold_number)


        lr = LinearRegression()
        lr.fit(fold_train_lr, fold_train_label_lr)
        predict_lr = lr.predict(fold_test_lr)

        svg_reg = SVR(kernel='rbf')
        svg_reg.fit(fold_train_svr, fold_train_label_svr)
        predict_svr= svg_reg.predict(fold_test_svr)

        rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
        rf_reg.fit(fold_train_rf, fold_train_label_rf)
        predict_rf= rf_reg.predict(fold_test_rf)

        """X = fold_test.copy()
        print(X)
        print(fold_train_label)
        X.reset_index(inplace=True)
        plt.scatter(X['index'], fold_test_label['total_cases'], color='red')
        plt.plot(X['index'], predict, color='blue')
        plt.show()"""

        difference_array_lr = np.subtract(predict_lr, fold_test_label_lr.to_numpy())
        difference_array_svr = np.subtract(predict_svr, fold_test_label_svr.to_numpy())
        difference_array_rf = np.subtract(predict_rf, fold_test_label_rf.to_numpy())
        squared_array_lr = np.square(difference_array_lr)
        squared_array_svr = np.square(difference_array_svr)
        squared_array_rf = np.square(difference_array_rf)
        mse_lr = mse_lr + squared_array_lr.mean()
        mse_svr = mse_svr + squared_array_svr.mean()
        mse_rf = mse_rf + squared_array_rf.mean()
    mse_rf=mse_rf/fold_number
    mse_lr=mse_lr/fold_number
    mse_svr=mse_svr/fold_number
    mse_total=mse_svr+mse_lr+mse_rf
    for i in range(3):
        if mse_rf<=mse_lr and mse_rf<=mse_svr and mse_svr<=mse_lr:
            coef_rf=mse_lr
            coef_lr=mse_rf
            coef_svr=mse_svr
        elif mse_rf<=mse_lr and mse_rf<=mse_svr and mse_lr<=mse_svr:
            coef_rf = mse_svr
            coef_svr=mse_rf
            coef_lr=mse_lr
        elif mse_lr <=mse_rf and mse_lr <=mse_svr and mse_rf <=mse_svr:
            coef_lr = mse_svr
            coef_svr = mse_lr
            coef_rf = mse_rf
        elif mse_lr <= mse_rf and mse_lr <= mse_svr and mse_svr <= mse_rf:
            coef_lr = mse_rf
            coef_rf = mse_lr
            coef_svr = mse_svr
        elif mse_svr <= mse_rf and mse_svr <= mse_lr and mse_rf <= mse_lr:
            coef_svr = mse_lr
            coef_lr = mse_svr
            coef_rf = mse_rf
        elif mse_svr <= mse_rf and mse_svr <= mse_lr and mse_lr <= mse_rf:
            coef_svr = mse_rf
            coef_rf = mse_svr
            coef_lr = mse_lr

    mse=0
    for i in range(1, fold_number + 1):
        fold_train_lr, fold_train_label_lr, fold_test_lr, fold_test_label_lr = partition_folds(train_dataset_lr,
                                                                                               train_dataset_label_lr,
                                                                                               i, fold_number)
        fold_train_svr, fold_train_label_svr, fold_test_svr, fold_test_label_svr = partition_folds(train_dataset_svr,
                                                                                                   train_dataset_label_svr,
                                                                                                   i, fold_number)
        fold_train_rf, fold_train_label_rf, fold_test_rf, fold_test_label_rf = partition_folds(train_dataset_rf,
                                                                                               train_dataset_label_rf,
                                                                                               i, fold_number)
        lr = LinearRegression()
        lr.fit(fold_train_lr, fold_train_label_lr)
        predict_l = lr.predict(fold_test_lr)
        predict_lr = []
        for item in predict_l:
            predict_lr.append(item[0])

        svg_reg = SVR(kernel='rbf')
        svg_reg.fit(fold_train_svr, fold_train_label_svr)
        predict_svr = svg_reg.predict(fold_test_svr)

        rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
        rf_reg.fit(fold_train_rf, fold_train_label_rf)
        predict_rf = rf_reg.predict(fold_test_rf)

        predict_lr_2 = np.divide( predict_lr,3)
        predict_rf_2 = np.divide( predict_rf,3)
        predict_svr_2 = np.divide( predict_svr,3)
        predict_2 = np.add(np.add(predict_lr_2, predict_rf_2), predict_svr_2)
        difference_array_2 = np.subtract(predict_2, fold_test_label_lr.to_numpy())
        squared_array_2= np.square(difference_array_2)
        mse_2= mse_2 + squared_array_2.mean()

        predict_lr=np.divide(np.multiply(coef_lr,predict_lr),mse_total)
        predict_rf=np.divide(np.multiply(coef_rf,predict_rf),mse_total)
        predict_svr=np.divide(np.multiply(coef_svr, predict_svr),mse_total)
        predict=np.add(np.add(predict_svr,predict_rf),predict_lr)
        difference_array = np.subtract(predict, fold_test_label_lr.to_numpy())
        squared_array = np.square(difference_array)
        mse = mse + squared_array.mean()

    print(mse/fold_number)
    return coef_lr,coef_svr,coef_rf,mse_total,(mse / fold_number)

def ensemble_prediction(train_dataset_lr,train_dataset_label_lr,train_dataset_svr,train_dataset_label_svr, train_dataset_rf,train_dataset_label_rf,test_dataset_lr,test_dataset_svr,test_dataset_rf):

    lr = LinearRegression()
    lr.fit(train_dataset_lr,train_dataset_label_lr)
    coef_lr,coef_svr,coef_rf,mse_total,mse=ensemble_model_evaluate_with_cross_validation(train_dataset_lr,train_dataset_label_lr,train_dataset_svr,train_dataset_label_svr,train_dataset_rf,train_dataset_label_rf,5)

    predict_l = lr.predict(test_dataset_lr)
    predict_lr=[]
    for item in predict_l:
        predict_lr.append(item[0])

    svg_reg = SVR(kernel='rbf')
    svg_reg.fit(train_dataset_svr, train_dataset_label_svr)
    predict_svr = svg_reg.predict(test_dataset_svr)
    rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
    rf_reg.fit(train_dataset_rf, train_dataset_label_rf)
    predict_rf = rf_reg.predict(test_dataset_rf)


    predict_lr = np.divide(np.multiply(coef_lr, predict_lr), mse_total)
    predict_rf = np.divide(np.multiply(coef_rf, predict_rf), mse_total)
    predict_svr = np.divide(np.multiply(coef_svr, predict_svr), mse_total)
    predict=np.add(np.add(predict_svr,predict_rf),predict_lr)
    print(predict)
    return predict
    """dosya = open('predictions.txt', 'a')
    dosya.write("\n"+str(predict))"""

def experiment_with_non_drop(model_list,generation_count,fold_number_x):
    for model in model_list:
        dosya = open('results10.txt', 'a')
        dosya2 = open('subsets10.txt', 'a')
        dosya.write(str(model)+" "+str(fold_number_x)+" drop " +str(generation_count)+" ")
        dosya2.write(str(model)+" "+str(fold_number_x)+" drop " +str(generation_count)+" ")
        print("Non-drop value "+ model)
        subset=subset_selection_with_genetic_algorithm(mean_dataset_train,mean_dataset_train_label,mean_dataset_train.columns,fold_number,5,generation_count,model)
        temp=mean_dataset_train.copy()
        from_backward,removed_features=backward_subset_selection(mean_dataset_train.copy(),mean_dataset_train_label,mean_dataset_train.columns,fold_number,model)
        backward_value=model_evaluate_with_cross_validation(from_backward,mean_dataset_train_label,fold_number,model)
        print("mean data after backward",backward_value)
        dataset_features = temp.columns.to_list()

        for i in range(len(subset)):
            if subset[i]==0:
                del temp[dataset_features[i]]

        dosya2.write("genetic "+str(temp.columns)+" "+str(len(temp.columns))+"\n")
        dosya2.write(str(model) + " " + str(fold_number_x) + " drop " + str(generation_count) + " ")
        dosya2.write("backward " + str(from_backward.columns)+" "+ str(len(from_backward.columns))+"\n")
        genetic_value=model_evaluate_with_cross_validation(temp,mean_dataset_train_label,fold_number,model)
        print("mean data after genetic",genetic_value)
        dosya.write(str(genetic_value) + " "+ str(backward_value) + "\n")
        dosya.close()
        dosya2.close()

def experiment_with_drop(model_list,generation_count,fold_number_x):

    for model in model_list:
        dosya = open('results10.txt', 'a')
        dosya2 = open('subsets10.txt', 'a')
        dosya.write("\n"+str(model)+" "+str(fold_number_x)+" non-drop " +str(generation_count)+" ")
        dosya2.write("\n"+str(model) + " " + str(fold_number_x) + " drop " + str(generation_count) + " ")
        print("Drop value " + model )
        subset = subset_selection_with_genetic_algorithm(drop_dataset_train, drop_dataset_train_label,
                                                                               drop_dataset_train.columns, fold_number, 5,
                                                                               generation_count,model)

        temp = drop_dataset_train.copy()
        from_backward, removed_features = backward_subset_selection(drop_dataset_train.copy(),drop_dataset_train_label,drop_dataset_train.columns,fold_number,model)
        backward_value=model_evaluate_with_cross_validation(from_backward, drop_dataset_train_label, fold_number,model)
        print("drop data after backward",backward_value)
        dataset_features = temp.columns.to_list()

        for i in range(len(subset)):

            if subset[i] == 0:
                del temp[dataset_features[i]]

        dosya2.write("genetic " + str(temp.columns) + " " + str(len(temp.columns)) + "\n")
        dosya2.write(str(model) + " " + str(fold_number_x) + " drop " + str(generation_count) + " ")
        dosya2.write("backward " + str(from_backward.columns) + " " + str(len(from_backward.columns)))
        genetic_value = model_evaluate_with_cross_validation(temp, drop_dataset_train_label, fold_number, model)
        print("mean data after genetic", genetic_value)
        dosya.write(str(genetic_value) + " " + str(backward_value) )
        dosya.close()
        dosya2.close()

"""lr = LinearRegression()
lr.fit(drop_dataset_train.iloc[0:1000,:], drop_dataset_train_label.iloc[0:1000,:])
predict = lr.predict(drop_dataset_train.iloc[1001:1198,:])
X=drop_dataset_train.copy()
X.reset_index(inplace=True)
X=X.iloc[1001:1198,:]
difference_array = np.subtract(predict, drop_dataset_train_label.iloc[1001:1198,:].to_numpy())
squared_array = np.square(difference_array)
mse =  squared_array.mean()
print(mse)
plt.scatter(X['index'], drop_dataset_train_label.iloc[1001:1198,:],color='red')
plt.plot(X['index'],predict,color='blue')
plt.show()"""

"""svg_reg=SVR(kernel='rbf')
svg_reg.fit(drop_dataset_train.iloc[0:1000,:],drop_dataset_train_label.iloc[0:1000,:])
predict = svg_reg.predict(drop_dataset_train.iloc[1001:1198,:])
X=drop_dataset_train.copy()
X.reset_index(inplace=True)
X=X.iloc[1001:1198,:]
difference_array = np.subtract(predict, drop_dataset_train_label.iloc[1001:1198,:].to_numpy())
squared_array = np.square(difference_array)
mse =  squared_array.mean()
print(mse)
plt.scatter(X['index'], drop_dataset_train_label.iloc[1001:1198,:],color='red')
plt.plot(X['index'],predict,color='blue')
plt.show()"""

"""r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(drop_dataset_train.iloc[0:1000,:],drop_dataset_train_label.iloc[0:1000,:])
predict = r_dt.predict(drop_dataset_train.iloc[1001:1198,:])
X=drop_dataset_train.copy()
X.reset_index(inplace=True)
X=X.iloc[1001:1198,:]
difference_array = np.subtract(predict, drop_dataset_train_label.iloc[1001:1198,:].to_numpy())
squared_array = np.square(difference_array)
mse =  squared_array.mean()
print(mse)
plt.scatter(X['index'], drop_dataset_train_label.iloc[1001:1198,:],color='red')
plt.plot(X['index'],predict,color='blue')
plt.show()"""

"""rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(drop_dataset_train.iloc[0:1000,:],drop_dataset_train_label.iloc[0:1000,:])
predict = rf_reg.predict(drop_dataset_train.iloc[1001:1198,:])
X=drop_dataset_train.copy()
X.reset_index(inplace=True)
X=X.iloc[1001:1198,:]
difference_array = np.subtract(predict, drop_dataset_train_label.iloc[1001:1198,:].to_numpy())
squared_array = np.square(difference_array)
mse =  squared_array.mean()
print(mse)
plt.scatter(X['index'], drop_dataset_train_label.iloc[1001:1198,:],color='red')
plt.plot(X['index'],predict,color='blue')
plt.show()"""


"""
model_list=['multiple_linear','svm','rf']
fold_number_list=[5]
generation_count_list=[5]
for fold_number_x in fold_number_list:

    for count in generation_count_list:
        fold_number=int(fold_number_x)
        experiment_with_drop(model_list,count,fold_number_x)
        experiment_with_non_drop(model_list,count,fold_number_x)
"""






"""

model_list=['multiple_linear','svm','rf']
subset_feature_MLR_mean=['city_x', 'ndvi_ne', 'ndvi_se', 'ndvi_sw', 'reanalysis_air_temp_k',
       'reanalysis_min_air_temp_k', 'reanalysis_relative_humidity_percent',
       'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
       'station_max_temp_c']

subset_feature_SVR_mean=['year_x', 'precipitation_amt_mm', 'reanalysis_min_air_temp_k',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',
       'station_max_temp_c']

subset_feature_RF_mean=['year_x', 'ndvi_ne', 'ndvi_nw', 'ndvi_se',
       'reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent',
       'reanalysis_specific_humidity_g_per_kg', 'station_diur_temp_rng_c',
       'station_min_temp_c']


subset_feature_MLR_drop=['city_x', 'ndvi_se', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
       'station_min_temp_c', 'station_precip_mm']
subset_feature_SVR_drop=['year_x', 'ndvi_se', 'precipitation_amt_mm',
       'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_min_temp_c']
subset_feature_RF_drop=['city_x', 'ndvi_ne', 'ndvi_nw', 'precipitation_amt_mm',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'station_avg_temp_c',
       'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']

all_features=mean_dataset_train.columns.tolist()
fold_number=5

for model in model_list:
    temp_train_data_drop=drop_dataset_train.copy()
    temp_train_data_mean=mean_dataset_train.copy()
    remained_features_drop=[]
    remained_features_mean=[]

    if model=='multiple_linear':
        remained_features_drop=[x for x in temp_train_data_drop.columns.tolist() if x not in subset_feature_MLR_drop]
        for feature in remained_features_drop:
            del temp_train_data_drop[feature]
        model_evaluate_with_cross_validation(temp_train_data_drop,drop_dataset_train_label,fold_number,model)
        remained_features_mean = [x for x in temp_train_data_mean.columns.tolist() if x not in subset_feature_MLR_mean]
        for feature in remained_features_mean:
            del temp_train_data_mean[feature]
        model_evaluate_with_cross_validation(temp_train_data_mean, mean_dataset_train_label, fold_number, model)

    elif model=='svm':
        remained_features_drop = [x for x in temp_train_data_drop.columns.tolist() if x not in subset_feature_SVR_drop]
        for feature in remained_features_drop:
            del temp_train_data_drop[feature]
        model_evaluate_with_cross_validation(temp_train_data_drop, drop_dataset_train_label, fold_number, model)
        remained_features_mean = [x for x in temp_train_data_mean.columns.tolist() if x not in subset_feature_SVR_mean]
        for feature in remained_features_mean:
            del temp_train_data_mean[feature]
        model_evaluate_with_cross_validation(temp_train_data_mean, mean_dataset_train_label, fold_number, model)

    elif model=='rf':
        remained_features_drop = [x for x in temp_train_data_drop.columns.tolist() if x not in subset_feature_RF_drop]
        for feature in remained_features_drop:
            del temp_train_data_drop[feature]
        model_evaluate_with_cross_validation(temp_train_data_drop, drop_dataset_train_label, fold_number, model)
        remained_features_mean = [x for x in temp_train_data_mean.columns.tolist() if x not in subset_feature_RF_mean]
        for feature in remained_features_mean:
            del temp_train_data_mean[feature]
        model_evaluate_with_cross_validation(temp_train_data_mean, mean_dataset_train_label, fold_number, model)



fold_number=5

temp_train_data_drop_lr=drop_dataset_train.copy()
temp_train_data_mean_lr=mean_dataset_train.copy()
temp_train_data_drop_svr=drop_dataset_train.copy()
temp_train_data_mean_svr=mean_dataset_train.copy()
temp_train_data_drop_rf=drop_dataset_train.copy()
temp_train_data_mean_rf=mean_dataset_train.copy()
remained_features_drop_lr=[x for x in temp_train_data_drop_lr.columns.tolist() if x not in subset_feature_MLR_drop]
for feature in remained_features_drop_lr:
    del temp_train_data_drop_lr[feature]
remained_features_mean_lr=[x for x in temp_train_data_mean_lr.columns.tolist() if x not in subset_feature_MLR_mean]
for feature in remained_features_mean_lr:
    del temp_train_data_mean_lr[feature]

remained_features_drop_svr=[x for x in temp_train_data_drop_svr.columns.tolist() if x not in subset_feature_SVR_drop]
for feature in remained_features_drop_svr:
    del temp_train_data_drop_svr[feature]
remained_features_mean_svr=[x for x in temp_train_data_mean_svr.columns.tolist() if x not in subset_feature_SVR_mean]
for feature in remained_features_mean_svr:
    del temp_train_data_mean_svr[feature]

remained_features_drop_rf=[x for x in temp_train_data_drop_rf.columns.tolist() if x not in subset_feature_RF_drop]
for feature in remained_features_drop_rf:
    del temp_train_data_drop_rf[feature]
remained_features_mean_rf=[x for x in temp_train_data_mean_rf.columns.tolist() if x not in subset_feature_RF_mean]
for feature in remained_features_mean_rf:
    del temp_train_data_mean_rf[feature]

print(ensemble_model_evaluate_with_cross_validation(temp_train_data_drop_lr,drop_dataset_train_label,temp_train_data_drop_svr,drop_dataset_train_label,temp_train_data_drop_rf,drop_dataset_train_label,fold_number))
print(ensemble_model_evaluate_with_cross_validation(temp_train_data_mean_lr,mean_dataset_train_label,temp_train_data_mean_svr,mean_dataset_train_label,temp_train_data_mean_rf,mean_dataset_train_label,fold_number))


#print(ensemble_model_evaluate_with_cross_validation(mean_dataset_train,mean_dataset_train_label,mean_dataset_train,mean_dataset_train_label,mean_dataset_train,mean_dataset_train_label,fold_number))



"""






model_list=['multiple_linear','svm','rf']
subset_feature_MLR_mean=['city_x', 'ndvi_ne', 'ndvi_se', 'ndvi_sw', 'reanalysis_air_temp_k',
       'reanalysis_min_air_temp_k', 'reanalysis_relative_humidity_percent',
       'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
       'station_max_temp_c']

subset_feature_SVR_mean=['year_x', 'precipitation_amt_mm', 'reanalysis_min_air_temp_k',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',
       'station_max_temp_c']

subset_feature_RF_mean=['year_x', 'ndvi_ne', 'ndvi_nw', 'ndvi_se',
       'reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent',
       'reanalysis_specific_humidity_g_per_kg', 'station_diur_temp_rng_c',
       'station_min_temp_c']


subset_feature_MLR_drop=['city_x', 'ndvi_se', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
       'station_min_temp_c', 'station_precip_mm']
subset_feature_SVR_drop=['year_x', 'ndvi_se', 'precipitation_amt_mm',
       'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_min_temp_c']
subset_feature_RF_drop=['city_x', 'ndvi_ne', 'ndvi_nw', 'precipitation_amt_mm',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'station_avg_temp_c',
       'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']
temp_train_data_drop_lr=drop_dataset_train.copy()
temp_train_data_drop_svr=drop_dataset_train.copy()
temp_train_data_drop_rf=drop_dataset_train.copy()
remained_features_drop_lr=[x for x in temp_train_data_drop_lr.columns.tolist() if x not in subset_feature_MLR_drop]
for feature in remained_features_drop_lr:
    del temp_train_data_drop_lr[feature]
remained_features_drop_svr=[x for x in temp_train_data_drop_svr.columns.tolist() if x not in subset_feature_SVR_drop]
for feature in remained_features_drop_svr:
    del temp_train_data_drop_svr[feature]
remained_features_drop_rf = [x for x in temp_train_data_drop_rf.columns.tolist() if x not in subset_feature_RF_drop]
for feature in remained_features_drop_rf:
    del temp_train_data_drop_rf[feature]








temp_test_data_lr=test_data.copy()
temp_test_data_svr=test_data.copy()
temp_test_data_rf=test_data.copy()
subset_feature_MLR_drop=['city', 'ndvi_se', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
       'station_min_temp_c', 'station_precip_mm']
subset_feature_SVR_drop=['year', 'ndvi_se', 'precipitation_amt_mm',
       'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_min_temp_c']
subset_feature_RF_drop=['city', 'ndvi_ne', 'ndvi_nw', 'precipitation_amt_mm',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'station_avg_temp_c',
       'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']

remained_features_drop_lr=[x for x in temp_test_data_lr.columns.tolist() if x not in subset_feature_MLR_drop]
for feature in remained_features_drop_lr:
    del temp_test_data_lr[feature]
remained_features_drop_svr=[x for x in temp_test_data_svr.columns.tolist() if x not in subset_feature_SVR_drop]
for feature in remained_features_drop_svr:
    del temp_test_data_svr[feature]
remained_features_drop_rf = [x for x in temp_test_data_rf.columns.tolist() if x not in subset_feature_RF_drop]
for feature in remained_features_drop_rf:
    del temp_test_data_rf[feature]
predict_11=ensemble_prediction(temp_train_data_drop_lr,drop_dataset_train_label,temp_train_data_drop_svr,drop_dataset_train_label,temp_train_data_drop_rf,drop_dataset_train_label,temp_test_data_lr,temp_test_data_svr,temp_test_data_rf)
predict_11=predict_11.astype(int)
np.savetxt('values23.csv',  predict_11 ,delimiter=",")






"""

subset_feature_SVR_drop=['city_x', 'year_x', 'ndvi_ne', 'ndvi_nw',
       'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k',
       'station_avg_temp_c']
temp_drop_dataset_train=drop_dataset_train.copy()
remained_features_drop=[x for x in temp_drop_dataset_train.columns.tolist() if x not in subset_feature_SVR_drop]
for feature in remained_features_drop:
    del temp_drop_dataset_train[feature]
subset_feature_SVR_drop_test=['city', 'year', 'ndvi_ne', 'ndvi_nw',
       'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k',
       'station_avg_temp_c']
temp_test_data=test_data.copy()
remained_features_drop_test=[x for x in test_data.columns.tolist() if x not in subset_feature_SVR_drop_test]
for feature in remained_features_drop_test:
    del temp_test_data[feature]

svg_reg = SVR(kernel='rbf')
svg_reg.fit(drop_dataset_train, drop_dataset_train_label)
predict_svr = svg_reg.predict(test_data)


temp_test_data.to_csv('array.csv', header=True, index=False)


subset_feature_SVR_drop_test=['city', 'year', 'weekofyear']
temp_test_data=test_data.copy()
remained_features_drop_test=[x for x in test_data.columns.tolist() if x not in subset_feature_SVR_drop_test]
for feature in remained_features_drop_test:
    del temp_test_data[feature]
predict_svr=predict_svr.astype(int)
np.savetxt('valuesssssss.csv',  predict_svr ,delimiter=",")
temp_test_data.loc[temp_test_data['city'] == 1, 'city'] = 'sj'
temp_test_data.loc[temp_test_data['city'] == 0, 'city'] = 'iq'


temp_test_data.loc[temp_test_data['year'] == 18, 'year'] = '2008'
temp_test_data.loc[temp_test_data['year'] == 19, 'year'] = '2009'
temp_test_data.loc[temp_test_data['year'] == 20, 'year'] = '2010'
temp_test_data.loc[temp_test_data['year'] == 21, 'year'] = '2011'
temp_test_data.loc[temp_test_data['year'] == 22, 'year'] = '2012'
temp_test_data.loc[temp_test_data['year'] == 23, 'year'] = '2013'

print(temp_test_data)
temp_test_data['total_cases']=predict_svr.astype(int)
temp_test_data.to_csv('arraysssss.csv', header=True, index=False)


"""