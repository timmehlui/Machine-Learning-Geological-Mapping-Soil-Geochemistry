# -*- coding: utf-8 -*-
"""
Created on 2022/03/15

@author: Timothy

Testing 8 sampling methods using a pipeline to correctly use cv and upscaling.
"""

# Import stuff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC #, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline
import pyrolite.comp
import datetime

from sameClassOrderNine import sameClassOrderNine
from mcsprob import mcsprob

# Process Data
seeds = list(range(0, 10))

for seed in seeds:
    # Load data, group adequately, and scale
    data = pd.read_csv("/Users/Timothy/Dropbox/GenFive Programming Files/dataMasterFiveNoTest.csv")
    data_full = pd.read_csv("/Users/Timothy/Dropbox/GenFive Programming Files/dataMasterFiveFull.csv")
    
    # Drop useless columns and highly correlated
    X = data.drop(['Easting', 'Northing', 'NGU_orig', 'OGU', 'InNewSet', 'NGU', 'Ca', 'Co', 'Cr', 'Fe'], axis=1)
    y = data['NGU']
    X_full = data_full.drop(['Easting', 'Northing', 'NGU_orig', 'OGU', 'InNewSet', 'NGU', 'Ca', 'Co', 'Cr', 'Fe'], axis=1)
    y_full = data_full['NGU']
    
    # Divide data sets into element part (for centered log ratio) and topographic part (for regular scaling)
    Xelems = X.drop(['SlopeRep', 'AspectRep', 'ElevRep'], axis=1) 
    Xelems_full = X_full.drop(['SlopeRep', 'AspectRep', 'ElevRep'], axis=1)
    Xtopo = X[['SlopeRep', 'AspectRep', 'ElevRep']].copy()
    Xtopo_full = X_full[['SlopeRep', 'AspectRep', 'ElevRep']].copy()
    
    # Centered log ratio is independent by soil sample, so doesn't need to be after train test split
    Xclr = Xelems.pyrocomp.CLR()
    Xclr_full = Xelems_full.pyrocomp.CLR()
    
    # Recombine with topographic features, to proceed with train test split
    Xrecomb = pd.concat([Xclr, Xtopo], axis=1)
    Xrecomb_full = pd.concat([Xclr_full, Xtopo_full], axis=1)
    
    X_train_og, X_test_og, y_train_og, y_test_og = train_test_split(Xrecomb, y, test_size=0.2, random_state=seed, stratify=y)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_og)
    X_test_sc = scaler.transform(X_test_og)
    X_full_sc = scaler.transform(Xrecomb_full)
    # Want it as a pandas dataframe, not numpy
    X_train_sc = pd.DataFrame(X_train_sc, columns=Xrecomb.columns)
    X_test_sc = pd.DataFrame(X_test_sc, columns=Xrecomb.columns)
    X_full_sc = pd.DataFrame(X_full_sc, columns=Xrecomb.columns)
    
    # Final train, test, and full sets
    X_train_final = X_train_sc.copy()
    X_test_final = X_test_sc.copy()
    X_full_final = X_full_sc.copy()
    y_train_final = y_train_og.copy()
    y_test_final = y_test_og.copy()
    y_full_final = y_full.copy()
    
    # Create different sampling methods
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    # 0: No change, stratified, imbalanced
    pipeline_setup0 = []
    
    # 1: Undersample to smallest size
    rus1 = RandomUnderSampler(random_state = seed)
    pipeline_setup1 = [('rus1', rus1)]

    # 2: Oversample to max
    ros2 = RandomOverSampler(random_state = seed)
    pipeline_setup2 = [('ros2', ros2)]
    
    # 3: SMOTE to max
    sm3 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    pipeline_setup3 = [('sm3', sm3)]
    
    # 4: ADASYN to max
    as4 = ADASYN(sampling_strategy = 'auto', n_neighbors = 5, random_state = seed)
    pipeline_setup4 = [('as4', as4)]
    
    # 5: SMOTEENN
    # SMOTE but then cleaning with Edited Nearest Neighbors
    smenn5 = SMOTEENN(sampling_strategy = 'auto', smote = sm3, random_state = seed)
    pipeline_setup5 = [('smenn5', smenn5)]
    
    # 6: SMOTETomek
    # SMOTE but then cleaning with Tomek
    smtomek6 = SMOTETomek(sampling_strategy = 'auto', smote = sm3, random_state = seed)
    pipeline_setup6 = [('smtomek6', smtomek6)]
    
    
    pipelines = [pipeline_setup0, pipeline_setup1, pipeline_setup2, pipeline_setup3,
                 pipeline_setup4, pipeline_setup5, pipeline_setup6]
    
    cv_results = [[], [], [], [], [], [], []]
    best_params = [[], [], [], [], [], [], []]
    summary_f1_macros_train = pd.DataFrame()
    summary_f1_macros_test = pd.DataFrame()
    summary_f1_macros_full = pd.DataFrame()
    
    #Change
    summary_f1_average_train = pd.DataFrame()
    summary_f1_average_test = pd.DataFrame()
    summary_f1_average_full = pd.DataFrame()
    
    # Loop over different pipelines of sampling methods
    for i in range(0, 7):
        print("Sampling Method")
        print(i)
        
        pipeline_setup = pipelines[i]
        n_iter_number = 100
        
        # Logistic Regression
        print("Logistic Regression")
        print(datetime.datetime.now().time())
        
        lr = LogisticRegression(solver = 'saga', random_state = seed)
            
        pipeline_lr = Pipeline(pipeline_setup + [('lr', lr)])
        param_grid_lr = [{'lr__penalty': ['l1', 'l2'],
                          'lr__C': np.power(10, np.arange(-3, 3, 0.25))}]
        gs_lr = RandomizedSearchCV(estimator = pipeline_lr, param_distributions = param_grid_lr, 
                                   cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_lr.fit(X_train_final, y_train_final)
        lr_pred_train = gs_lr.predict(X_train_final)
        lr_pred_test = gs_lr.predict(X_test_final)
        lr_pred_full = gs_lr.predict(X_full_final)
        lr_f1_train_report = classification_report(y_train_final, lr_pred_train, output_dict=True)
        lr_f1_test_report = classification_report(y_test_final, lr_pred_test, output_dict=True)
        lr_f1_full_report = classification_report(y_full_final, lr_pred_full, output_dict=True)
        lr_probs_train = gs_lr.predict_proba(X_train_final)
        lr_probs_test = gs_lr.predict_proba(X_test_final)
        lr_probs_full = gs_lr.predict_proba(X_full_final)
        lr_best_params = gs_lr.best_params_
        cv_results[i].append(gs_lr.cv_results_)        
        
        # Quadratic Discriminant Analysis
        print("Quadratic Discriminant Analysis")
        print(datetime.datetime.now().time())
        
        qda = QuadraticDiscriminantAnalysis()
            
        pipeline_qda = Pipeline(pipeline_setup + [('qda', qda)])
        param_grid_qda = [{'qda__tol': np.power(10, np.arange(-9, -5, 0.25)),
                           'qda__reg_param': np.power(10, np.arange(-3, 3, 0.25))}]
        gs_qda = RandomizedSearchCV(estimator = pipeline_qda, param_distributions = param_grid_qda, 
                                    cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_qda.fit(X_train_final, y_train_final)
        qda_pred_train = gs_qda.predict(X_train_final)
        qda_pred_test = gs_qda.predict(X_test_final)
        qda_pred_full = gs_qda.predict(X_full_final)
        qda_f1_train_report = classification_report(y_train_final, qda_pred_train, output_dict=True)
        qda_f1_test_report = classification_report(y_test_final, qda_pred_test, output_dict=True)
        qda_f1_full_report = classification_report(y_full_final, qda_pred_full, output_dict=True)
        qda_probs_train = gs_qda.predict_proba(X_train_final)
        qda_probs_test = gs_qda.predict_proba(X_test_final)
        qda_probs_full = gs_qda.predict_proba(X_full_final)
        qda_best_params = gs_qda.best_params_
        cv_results[i].append(gs_qda.cv_results_)    
        
        # Nearest Neighbors
        print("Nearest Neighbors")
        print(datetime.datetime.now().time())
        
        nn = KNeighborsClassifier(algorithm = 'auto', weights = 'distance')
            
        pipeline_nn = Pipeline(pipeline_setup + [('nn', nn)])
        param_grid_nn = [{'nn__n_neighbors': np.arange(2, 9),
                          'nn__weights': ['uniform', 'distance'],
                          'nn__algorithm': ['ball_tree', 'kd_tree'],
                          'nn__leaf_size': [2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 70]}]
        gs_nn = RandomizedSearchCV(estimator = pipeline_nn, param_distributions = param_grid_nn, 
                                   cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_nn.fit(X_train_final, y_train_final)
        nn_pred_train = gs_nn.predict(X_train_final)
        nn_pred_test = gs_nn.predict(X_test_final)
        nn_pred_full = gs_nn.predict(X_full_final)
        nn_f1_train_report = classification_report(y_train_final, nn_pred_train, output_dict=True)
        nn_f1_test_report = classification_report(y_test_final, nn_pred_test, output_dict=True)
        nn_f1_full_report = classification_report(y_full_final, nn_pred_full, output_dict=True)
        nn_probs_train = gs_nn.predict_proba(X_train_final)
        nn_probs_test = gs_nn.predict_proba(X_test_final)
        nn_probs_full = gs_nn.predict_proba(X_full_final)
        nn_best_params = gs_nn.best_params_
        cv_results[i].append(gs_nn.cv_results_)    
        
        # Radial Basis Function Support-Vector Machine
        print("Radial Basis Function Support-Vector Machine")
        print(datetime.datetime.now().time())
        
        rbfsvm = SVC(kernel = 'rbf', probability = True, random_state = seed)
            
        pipeline_rbfsvm = Pipeline(pipeline_setup + [('rbfsvm', rbfsvm)])
        param_grid_rbfsvm = [{'rbfsvm__C': np.power(10, np.arange(-3, 3, 0.25)),
                              'rbfsvm__gamma': np.power(10, np.arange(-3, 3, 0.25)),
                              'rbfsvm__tol': np.power(10, np.arange(-10, -6, 0.5))}]
        gs_rbfsvm = RandomizedSearchCV(estimator = pipeline_rbfsvm, param_distributions = param_grid_rbfsvm, 
                                       cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_rbfsvm.fit(X_train_final, y_train_final)
        rbfsvm_pred_train = gs_rbfsvm.predict(X_train_final)
        rbfsvm_pred_test = gs_rbfsvm.predict(X_test_final)
        rbfsvm_pred_full = gs_rbfsvm.predict(X_full_final)
        rbfsvm_f1_train_report = classification_report(y_train_final, rbfsvm_pred_train, output_dict=True)
        rbfsvm_f1_test_report = classification_report(y_test_final, rbfsvm_pred_test, output_dict=True)
        rbfsvm_f1_full_report = classification_report(y_full_final, rbfsvm_pred_full, output_dict=True)
        rbfsvm_probs_train = gs_rbfsvm.predict_proba(X_train_final)
        rbfsvm_probs_test = gs_rbfsvm.predict_proba(X_test_final)
        rbfsvm_probs_full = gs_rbfsvm.predict_proba(X_full_final)
        rbfsvm_best_params = gs_rbfsvm.best_params_
        cv_results[i].append(gs_rbfsvm.cv_results_)
        
        # Naive Bayes
        print("Naive Bayes")
        print(datetime.datetime.now().time())
        
        nb = GaussianNB()
            
        pipeline_nb = Pipeline(pipeline_setup + [('nb', nb)])
        param_grid_nb = [{'nb__var_smoothing': np.power(10, np.arange(-20, -10, 0.25))}]
        gs_nb = RandomizedSearchCV(estimator = pipeline_nb, param_distributions = param_grid_nb, 
                                   cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_nb.fit(X_train_final, y_train_final)
        nb_pred_train = gs_nb.predict(X_train_final)
        nb_pred_test = gs_nb.predict(X_test_final)
        nb_pred_full = gs_nb.predict(X_full_final)
        nb_f1_train_report = classification_report(y_train_final, nb_pred_train, output_dict=True)
        nb_f1_test_report = classification_report(y_test_final, nb_pred_test, output_dict=True)
        nb_f1_full_report = classification_report(y_full_final, nb_pred_full, output_dict=True)
        nb_probs_train = gs_nb.predict_proba(X_train_final)
        nb_probs_test = gs_nb.predict_proba(X_test_final)
        nb_probs_full = gs_nb.predict_proba(X_full_final)
        nb_best_params = gs_nb.best_params_
        cv_results[i].append(gs_nb.cv_results_)    
        
        # Artificial Neural Network
        print("Artificial Neural Network")
        print(datetime.datetime.now().time())
        
        ann = MLPClassifier(random_state = seed, max_iter = 1000)
            
        pipeline_ann = Pipeline(pipeline_setup + [('ann', ann)])
        param_grid_ann = [{'ann__hidden_layer_sizes': [(15), (16), (17), (18), (19), (20), (21), (22), (23), (24), (25), (26), (27), (28), (29), (30), (31),
                                                       (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21),
                                                       (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31),
                                                       (15, 15, 15), (16, 16, 16), (17, 17, 17), (18, 18, 18), (19, 19, 19), (20, 20, 20), (21, 21, 21), (22, 22, 22),
                                                       (23, 23, 23), (24, 24, 24), (25, 25, 25), (26, 26, 26), (27, 27, 27), (28, 28, 28), (29, 29, 29), (30, 30, 30), (31, 31, 31)],
                           'ann__alpha': np.power(10, np.arange(-4, 2, 0.5)),
                           'ann__activation': ['identity', 'logistic', 'tanh', 'relu']}]
        gs_ann = RandomizedSearchCV(estimator = pipeline_ann, param_distributions = param_grid_ann, 
                                    cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_ann.fit(X_train_final, y_train_final)
        ann_pred_train = gs_ann.predict(X_train_final)
        ann_pred_test = gs_ann.predict(X_test_final)
        ann_pred_full = gs_ann.predict(X_full_final)
        ann_f1_train_report = classification_report(y_train_final, ann_pred_train, output_dict=True)
        ann_f1_test_report = classification_report(y_test_final, ann_pred_test, output_dict=True)
        ann_f1_full_report = classification_report(y_full_final, ann_pred_full, output_dict=True)
        ann_probs_train = gs_ann.predict_proba(X_train_final)
        ann_probs_test = gs_ann.predict_proba(X_test_final)
        ann_probs_full = gs_ann.predict_proba(X_full_final)
        ann_best_params = gs_ann.best_params_
        cv_results[i].append(gs_ann.cv_results_)    
        
        # Random Forest
        print("Random Forest")
        print(datetime.datetime.now().time())
        
        rf = RandomForestClassifier(random_state = seed)
            
        pipeline_rf = Pipeline(pipeline_setup + [('rf', rf)])
        param_grid_rf = [{'rf__max_depth': [10, 20, 30, 40, 50, 75, 100, 200, 400, 800, 1000],
                          'rf__min_samples_split': np.arange(2, 10),
                          'rf__n_estimators': [10, 20, 30, 40, 50, 75, 100, 200, 400, 800, 1000],
                          'rf__criterion': ['gini', 'entropy']}]
        gs_rf = RandomizedSearchCV(estimator = pipeline_rf, param_distributions = param_grid_rf, 
                                   cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_rf.fit(X_train_final, y_train_final)
        rf_pred_train = gs_rf.predict(X_train_final)
        rf_pred_test = gs_rf.predict(X_test_final)
        rf_pred_full = gs_rf.predict(X_full_final)
        rf_f1_train_report = classification_report(y_train_final, rf_pred_train, output_dict=True)
        rf_f1_test_report = classification_report(y_test_final, rf_pred_test, output_dict=True)
        rf_f1_full_report = classification_report(y_full_final, rf_pred_full, output_dict=True)
        rf_probs_train = gs_rf.predict_proba(X_train_final)
        rf_probs_test = gs_rf.predict_proba(X_test_final)
        rf_probs_full = gs_rf.predict_proba(X_full_final)
        rf_best_params = gs_rf.best_params_
        cv_results[i].append(gs_rf.cv_results_)    
        
        # AdaBoost Random Forest
        print("AdaBoost Random Forest")
        print(datetime.datetime.now().time())
        
        ab = AdaBoostClassifier(random_state = seed)

        pipeline_ab = Pipeline(pipeline_setup + [('ab', ab)])
        param_grid_ab = [{'ab__learning_rate': [2, 3, 4, 6, 10, 20, 50],
                          'ab__n_estimators': [10, 20, 30, 40, 50, 75, 100, 200, 400, 800, 1000, 1500, 2000],
                          'ab__base_estimator': [DecisionTreeClassifier(max_depth=1),
                                                 DecisionTreeClassifier(max_depth=2),
                                                 DecisionTreeClassifier(max_depth=3),
                                                 DecisionTreeClassifier(max_depth=4),
                                                 DecisionTreeClassifier(max_depth=5),
                                                 DecisionTreeClassifier(max_depth=6),
                                                 DecisionTreeClassifier(max_depth=7),
                                                 DecisionTreeClassifier(max_depth=10),
                                                 DecisionTreeClassifier(max_depth=13),
                                                 DecisionTreeClassifier(max_depth=15),
                                                 DecisionTreeClassifier(max_depth=18),
                                                 DecisionTreeClassifier(max_depth=20)]}]
        gs_ab = RandomizedSearchCV(estimator = pipeline_ab, param_distributions = param_grid_ab, 
                                   cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_ab.fit(X_train_final, y_train_final)
        ab_pred_train = gs_ab.predict(X_train_final)
        ab_pred_test = gs_ab.predict(X_test_final)
        ab_pred_full = gs_ab.predict(X_full_final)
        ab_f1_train_report = classification_report(y_train_final, ab_pred_train, output_dict=True)
        ab_f1_test_report = classification_report(y_test_final, ab_pred_test, output_dict=True)
        ab_f1_full_report = classification_report(y_full_final, ab_pred_full, output_dict=True)
        ab_probs_train = gs_ab.predict_proba(X_train_final)
        ab_probs_test = gs_ab.predict_proba(X_test_final)
        ab_probs_full = gs_ab.predict_proba(X_full_final)
        ab_best_params = gs_ab.best_params_
        cv_results[i].append(gs_ab.cv_results_)    
        
        # Gradient Boosting Random Forest
        print("Gradient Boosting Random Forest")
        print(datetime.datetime.now().time())
        
        gb = GradientBoostingClassifier(random_state = seed)
            
        pipeline_gb = Pipeline(pipeline_setup + [('gb', gb)])
        param_grid_gb = [{'gb__n_estimators': [10, 20, 30, 40, 50, 75, 100, 200, 400, 800],
                          'gb__learning_rate': np.power(10, np.arange(-4, 2, 0.5)),
                          'gb__min_samples_split': np.arange(2, 10),
                          'gb__min_samples_leaf': np.arange(2, 10),
                          'gb__max_depth': np.arange(2, 6)}]
        gs_gb = RandomizedSearchCV(estimator = pipeline_gb, param_distributions = param_grid_gb, 
                                   cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro', n_iter = n_iter_number)
        gs_gb.fit(X_train_final, y_train_final)
        gb_pred_train = gs_gb.predict(X_train_final)
        gb_pred_test = gs_gb.predict(X_test_final)
        gb_pred_full = gs_gb.predict(X_full_final)
        gb_f1_train_report = classification_report(y_train_final, gb_pred_train, output_dict=True)
        gb_f1_test_report = classification_report(y_test_final, gb_pred_test, output_dict=True)
        gb_f1_full_report = classification_report(y_full_final, gb_pred_full, output_dict=True)
        gb_probs_train = gs_gb.predict_proba(X_train_final)
        gb_probs_test = gs_gb.predict_proba(X_test_final)
        gb_probs_full = gs_gb.predict_proba(X_full_final)
        gb_best_params = gs_gb.best_params_
        cv_results[i].append(gs_gb.cv_results_)    
        
        # Create MCS models
        # MCS models
        # Double check to make sure all class orders are the same before creating the multi classifier system (MCS)
        matchingClassOrder = sameClassOrderNine(gs_lr, gs_qda, gs_nn, gs_rbfsvm, gs_nb, gs_ann, gs_rf, gs_ab, gs_gb)
        
        # Create the multi classifier systems (MCS)
        # Note description of MCS still says LSVM, but everytime it gets mentioned
        # it is actually not there
        
        classOrder = gs_lr.classes_
        if matchingClassOrder:
            # MCS with 2: ann, gb (>75%) with no repetition of forest method
            mcs2predtrain = mcsprob(classOrder, ann_probs_train, gb_probs_train)
            mcs2train_report = classification_report(y_train_og, mcs2predtrain, output_dict=True)
            # MCS with top 3, ann, rf, gb (>75%)
            mcs3predtrain = mcsprob(classOrder, ann_probs_train, rf_probs_train, gb_probs_train)
            mcs3train_report = classification_report(y_train_og, mcs3predtrain, output_dict=True)
            # MCS with 5: lr, nn, lsvm, ann, gb (>50%) with no repetition of forests or SVM
            mcs5predtrain = mcsprob(classOrder, lr_probs_train, nn_probs_train, ann_probs_train, gb_probs_train)
            mcs5train_report = classification_report(y_train_og, mcs5predtrain, output_dict=True)
            # MCS with top 6, lr, nn, lsvm, rbfsvm, ann, gb (>50%) with no repetition of forests
            mcs6predtrain = mcsprob(classOrder, lr_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, gb_probs_train)
            mcs6train_report = classification_report(y_train_og, mcs6predtrain, output_dict=True)
            # MCS with top 7, lr, nn, lsvm, ann, rf, ab, gb (>50%) with no repetition of SVM
            mcs7predtrain = mcsprob(classOrder, lr_probs_train, nn_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
            mcs7train_report = classification_report(y_train_og, mcs7predtrain, output_dict=True)
            # MCS with 8, remove rf and ab to have no repetition of forests
            mcs8predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, nb_probs_train, ann_probs_train, gb_probs_train)
            mcs8train_report = classification_report(y_train_og, mcs8predtrain, output_dict=True)
            # MCS with 9, remove nb for being too low. (>40%)
            mcs9predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
            mcs9train_report = classification_report(y_train_og, mcs9predtrain, output_dict=True)
            # MCS with all 10
            mcs10predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, nb_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
            mcs10train_report = classification_report(y_train_og, mcs10predtrain, output_dict=True)
            
            # MCS with 2: ann, gb (>75%) with no repetition of forest method
            mcs2predtest = mcsprob(classOrder, ann_probs_test, gb_probs_test)
            mcs2test_report = classification_report(y_test_og, mcs2predtest, output_dict=True)
            # MCS with top 3, ann, rf, gb (>75%)
            mcs3predtest = mcsprob(classOrder, ann_probs_test, rf_probs_test, gb_probs_test)
            mcs3test_report = classification_report(y_test_og, mcs3predtest, output_dict=True)
            # MCS with 5: lr, nn, lsvm, ann, gb (>50%) with no repetition of forests or SVM
            mcs5predtest = mcsprob(classOrder, lr_probs_test, nn_probs_test, ann_probs_test, gb_probs_test)
            mcs5test_report = classification_report(y_test_og, mcs5predtest, output_dict=True)
            # MCS with top 6, lr, nn, lsvm, rbfsvm, ann, gb (>50%) with no repetition of forests
            mcs6predtest = mcsprob(classOrder, lr_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, gb_probs_test)
            mcs6test_report = classification_report(y_test_og, mcs6predtest, output_dict=True)
            # MCS with top 7, lr, nn, lsvm, ann, rf, ab, gb (>50%) with no repetition of SVM
            mcs7predtest = mcsprob(classOrder, lr_probs_test, nn_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
            mcs7test_report = classification_report(y_test_og, mcs7predtest, output_dict=True)
            # MCS with 8, remove rf and ab to have no repetition of forests
            mcs8predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, nb_probs_test, ann_probs_test, gb_probs_test)
            mcs8test_report = classification_report(y_test_og, mcs8predtest, output_dict=True)
            # MCS with 9, remove nb for being too low. (>40%)
            mcs9predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
            mcs9test_report = classification_report(y_test_og, mcs9predtest, output_dict=True)
            # MCS with all 10
            mcs10predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, nb_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
            mcs10test_report = classification_report(y_test_og, mcs10predtest, output_dict=True)
            
            # MCS2 with full data set
            mcs2predfull = mcsprob(classOrder, ann_probs_full, rf_probs_full, gb_probs_full)
            mcs2full_report = classification_report(y_full, mcs2predfull, output_dict=True)
            # MCS3 with full data set
            mcs3predfull = mcsprob(classOrder, ann_probs_full, rf_probs_full, gb_probs_full)
            mcs3full_report = classification_report(y_full, mcs3predfull, output_dict=True)
            # MCS5 with full data set
            mcs5predfull = mcsprob(classOrder, lr_probs_full, nn_probs_full, ann_probs_full, gb_probs_full)
            mcs5full_report = classification_report(y_full, mcs5predfull, output_dict=True)
            # MCS6 with full data set
            mcs6predfull = mcsprob(classOrder, lr_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, gb_probs_full)
            mcs6full_report = classification_report(y_full, mcs6predfull, output_dict=True)
            # MCS7 with full data set
            mcs7predfull = mcsprob(classOrder, lr_probs_full, nn_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
            mcs7full_report = classification_report(y_full, mcs7predfull, output_dict=True)
            # MCS8 with full data set
            mcs8predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, nb_probs_full, ann_probs_full, gb_probs_full)
            mcs8full_report = classification_report(y_full, mcs8predfull, output_dict=True)
            # MCS9 with full data set
            mcs9predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
            mcs9full_report = classification_report(y_full, mcs9predfull, output_dict=True)
            # MCS10 with full data set
            mcs10predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, nb_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
            mcs10full_report = classification_report(y_full, mcs10predfull, output_dict=True)
        
        else:
            print('Failed matching class order')
        
        # Create the trivial classifier
        # The most common unit is Whitehorse
        s = pd.Series(['Whitehorse'])
        trivialpredtrain = s.repeat([len(y_train_og)])
        trivialpredtest = s.repeat([len(y_test_og)])
        trivialpredfull = s.repeat([len(y_full)])
        
        trivialtrain_report = classification_report(y_train_og, trivialpredtrain, output_dict=True)
        trivialtest_report = classification_report(y_test_og, trivialpredtest, output_dict=True)
        trivialfull_report = classification_report(y_full, trivialpredfull, output_dict=True)
        

        # Save F1 Macro Data
        lr_dict_train = pd.DataFrame.from_dict(lr_f1_train_report)
        qda_dict_train = pd.DataFrame.from_dict(qda_f1_train_report)
        nn_dict_train = pd.DataFrame.from_dict(nn_f1_train_report)
        rbfsvm_dict_train = pd.DataFrame.from_dict(rbfsvm_f1_train_report)
        nb_dict_train = pd.DataFrame.from_dict(nb_f1_train_report)
        ann_dict_train = pd.DataFrame.from_dict(ann_f1_train_report)
        rf_dict_train = pd.DataFrame.from_dict(rf_f1_train_report)
        ab_dict_train = pd.DataFrame.from_dict(ab_f1_train_report)
        gb_dict_train = pd.DataFrame.from_dict(gb_f1_train_report)
        mcs2_dict_train = pd.DataFrame.from_dict(mcs2train_report)
        mcs3_dict_train = pd.DataFrame.from_dict(mcs3train_report)
        mcs5_dict_train = pd.DataFrame.from_dict(mcs5train_report)
        mcs6_dict_train = pd.DataFrame.from_dict(mcs6train_report)
        mcs7_dict_train = pd.DataFrame.from_dict(mcs7train_report)
        mcs8_dict_train = pd.DataFrame.from_dict(mcs8train_report)
        mcs9_dict_train = pd.DataFrame.from_dict(mcs9train_report)
        mcs10_dict_train = pd.DataFrame.from_dict(mcs10train_report)
        trivial_dict_train = pd.DataFrame.from_dict(trivialtrain_report)
        
        lr_dict_test = pd.DataFrame.from_dict(lr_f1_test_report)
        qda_dict_test = pd.DataFrame.from_dict(qda_f1_test_report)
        nn_dict_test = pd.DataFrame.from_dict(nn_f1_test_report)
        rbfsvm_dict_test = pd.DataFrame.from_dict(rbfsvm_f1_test_report)
        nb_dict_test = pd.DataFrame.from_dict(nb_f1_test_report)
        ann_dict_test = pd.DataFrame.from_dict(ann_f1_test_report)
        rf_dict_test = pd.DataFrame.from_dict(rf_f1_test_report)
        ab_dict_test = pd.DataFrame.from_dict(ab_f1_test_report)
        gb_dict_test = pd.DataFrame.from_dict(gb_f1_test_report)
        mcs2_dict_test = pd.DataFrame.from_dict(mcs2test_report)
        mcs3_dict_test = pd.DataFrame.from_dict(mcs3test_report)
        mcs5_dict_test = pd.DataFrame.from_dict(mcs5test_report)
        mcs6_dict_test = pd.DataFrame.from_dict(mcs6test_report)
        mcs7_dict_test = pd.DataFrame.from_dict(mcs7test_report)
        mcs8_dict_test = pd.DataFrame.from_dict(mcs8test_report)
        mcs9_dict_test = pd.DataFrame.from_dict(mcs9test_report)
        mcs10_dict_test = pd.DataFrame.from_dict(mcs10test_report)
        trivial_dict_test = pd.DataFrame.from_dict(trivialtest_report)
        
        lr_dict_full = pd.DataFrame.from_dict(lr_f1_full_report)
        qda_dict_full = pd.DataFrame.from_dict(qda_f1_full_report)
        nn_dict_full = pd.DataFrame.from_dict(nn_f1_full_report)
        rbfsvm_dict_full = pd.DataFrame.from_dict(rbfsvm_f1_full_report)
        nb_dict_full = pd.DataFrame.from_dict(nb_f1_full_report)
        ann_dict_full = pd.DataFrame.from_dict(ann_f1_full_report)
        rf_dict_full = pd.DataFrame.from_dict(rf_f1_full_report)
        ab_dict_full = pd.DataFrame.from_dict(ab_f1_full_report)
        gb_dict_full = pd.DataFrame.from_dict(gb_f1_full_report)
        mcs2_dict_full = pd.DataFrame.from_dict(mcs2full_report)
        mcs3_dict_full = pd.DataFrame.from_dict(mcs3full_report)
        mcs5_dict_full = pd.DataFrame.from_dict(mcs5full_report)
        mcs6_dict_full = pd.DataFrame.from_dict(mcs6full_report)
        mcs7_dict_full = pd.DataFrame.from_dict(mcs7full_report)
        mcs8_dict_full = pd.DataFrame.from_dict(mcs8full_report)
        mcs9_dict_full = pd.DataFrame.from_dict(mcs9full_report)
        mcs10_dict_full = pd.DataFrame.from_dict(mcs10full_report)
        trivial_dict_full = pd.DataFrame.from_dict(trivialfull_report)
        
        f1_matrix_train = pd.concat([lr_dict_train.transpose()['f1-score'], qda_dict_train.transpose()['f1-score'],
                       nn_dict_train.transpose()['f1-score'], rbfsvm_dict_train.transpose()['f1-score'], 
                       nb_dict_train.transpose()['f1-score'], ann_dict_train.transpose()['f1-score'], 
                       rf_dict_train.transpose()['f1-score'], ab_dict_train.transpose()['f1-score'], 
                       gb_dict_train.transpose()['f1-score'],
                       mcs2_dict_train.transpose()['f1-score'], mcs3_dict_train.transpose()['f1-score'],
                       mcs5_dict_train.transpose()['f1-score'], mcs6_dict_train.transpose()['f1-score'],
                       mcs7_dict_train.transpose()['f1-score'], mcs8_dict_train.transpose()['f1-score'],
                       mcs9_dict_train.transpose()['f1-score'], mcs10_dict_train.transpose()['f1-score'],
                       trivial_dict_train.transpose()['f1-score']], axis=1)
    
        f1_matrix_test = pd.concat([lr_dict_test.transpose()['f1-score'], qda_dict_test.transpose()['f1-score'],
                       nn_dict_test.transpose()['f1-score'], rbfsvm_dict_test.transpose()['f1-score'], 
                       nb_dict_test.transpose()['f1-score'], ann_dict_test.transpose()['f1-score'], 
                       rf_dict_test.transpose()['f1-score'], ab_dict_test.transpose()['f1-score'], 
                       gb_dict_test.transpose()['f1-score'],
                       mcs2_dict_test.transpose()['f1-score'], mcs3_dict_test.transpose()['f1-score'],
                       mcs5_dict_test.transpose()['f1-score'], mcs6_dict_test.transpose()['f1-score'],
                       mcs7_dict_test.transpose()['f1-score'], mcs8_dict_test.transpose()['f1-score'],
                       mcs9_dict_test.transpose()['f1-score'], mcs10_dict_test.transpose()['f1-score'],
                       trivial_dict_test.transpose()['f1-score']], axis=1)
    
        f1_matrix_full = pd.concat([lr_dict_full.transpose()['f1-score'], qda_dict_full.transpose()['f1-score'],
                       nn_dict_full.transpose()['f1-score'], rbfsvm_dict_full.transpose()['f1-score'], 
                       nb_dict_full.transpose()['f1-score'], ann_dict_full.transpose()['f1-score'], 
                       rf_dict_full.transpose()['f1-score'], ab_dict_full.transpose()['f1-score'], 
                       gb_dict_full.transpose()['f1-score'],
                       mcs2_dict_full.transpose()['f1-score'], mcs3_dict_full.transpose()['f1-score'],
                       mcs5_dict_full.transpose()['f1-score'], mcs6_dict_full.transpose()['f1-score'],
                       mcs7_dict_full.transpose()['f1-score'], mcs8_dict_full.transpose()['f1-score'],
                       mcs9_dict_full.transpose()['f1-score'], mcs10_dict_full.transpose()['f1-score'],
                       trivial_dict_full.transpose()['f1-score']], axis=1)
    
        summary_f1_macros_train = pd.concat([summary_f1_macros_train, f1_matrix_train.iloc[[6]]], axis = 0)
        summary_f1_macros_test = pd.concat([summary_f1_macros_test, f1_matrix_test.iloc[[6]]], axis = 0)
        summary_f1_macros_full = pd.concat([summary_f1_macros_full, f1_matrix_full.iloc[[6]]], axis = 0)
        
        # change
        summary_f1_average_train = pd.concat([summary_f1_average_train, f1_matrix_train.iloc[[5]]], axis = 0)
        summary_f1_average_test = pd.concat([summary_f1_average_test, f1_matrix_test.iloc[[5]]], axis = 0)
        summary_f1_average_full = pd.concat([summary_f1_average_full, f1_matrix_full.iloc[[5]]], axis = 0)
    
        # Save Hyperparameter Data
        best_params[i].append(lr_best_params)
        best_params[i].append(qda_best_params)
        best_params[i].append(nn_best_params)
        best_params[i].append(rbfsvm_best_params)
        best_params[i].append(nb_best_params)
        best_params[i].append(ann_best_params)
        best_params[i].append(rf_best_params)
        best_params[i].append(ab_best_params)
        best_params[i].append(gb_best_params)
        
        dfcv = pd.DataFrame(cv_results)
        dfbp = pd.DataFrame(best_params)
        
        dfcv.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1dfcvsamp'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        f1_matrix_train.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1f1trainsamp'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        f1_matrix_test.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1f1testsamp'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        f1_matrix_full.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1f1fullsamp'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        dfbp.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g5v1dfbpsamp'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
    
    summary_f1_macros_train.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1summarytrainseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_macros_test.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1summarytestseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_macros_full.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1summaryfullseed'+str(seed)+'.csv', index=None, header = True)
    
    #change
    summary_f1_average_train.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1averagesummarytrainseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_average_test.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1averagesummarytestseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_average_full.to_csv(r'C:\Users\Timothy\Dropbox\GenSeven Programming Files\SamplingMethodsResults\g7v1averagesummaryfullseed'+str(seed)+'.csv', index=None, header = True)