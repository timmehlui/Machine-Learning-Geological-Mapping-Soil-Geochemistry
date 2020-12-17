# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:22:48 2020

@author: Timothy

Test whether topographical data helps or not.
"""
# Import stuff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
import datetime

seeds = [720, 721, 722, 723, 724, 725, 726, 727, 728, 729]

for seed in seeds:
    # Load data, group adequately, and scale
    data = pd.read_csv("/Users/Timothy/Dropbox/Undergrad Thesis Backups/Programming Files Dropbox/dataMasterNoSimpson.csv")
    data_full = pd.read_csv("/Users/Timothy/Dropbox/Undergrad Thesis Backups/Programming Files Dropbox/dataMasterCleanedHasTestNoSimpson.csv")
    
    # Dimensionality reduction schemes
    # Test 0: With topographic data
    X0 = np.array(data.drop(['Easting', 'Northing', 'GeoUnit', 'Al', 'K', 'Na', 'Sr'], axis=1))
    y0 = np.array(data['GeoUnit'])
    X0_full = np.array(data_full.drop(['Easting', 'Northing', 'GeoUnit', 'Al', 'K', 'Na', 'Sr'], axis=1))
    y0_full = np.array(data_full['GeoUnit'])
    # Test 1: Without topographic data
    X1 = np.array(data.drop(['Easting', 'Northing', 'GeoUnit', 'SlopeRep', 'AspectRep', 'Al', 'K', 'Na', 'Sr'], axis=1))
    y1 = np.array(data['GeoUnit'])
    X1_full = np.array(data_full.drop(['Easting', 'Northing', 'GeoUnit', 'SlopeRepr', 'AspectRep', 'Al', 'K', 'Na', 'Sr'], axis=1))
    y1_full = np.array(data_full['GeoUnit'])
    
    data_sets = [[X0, y0, X0_full, y0_full], [X1, y1, X1_full, y1_full]]
    
    cv_results = [[], []]
    best_params = [[], []]
    summary_f1_macros_train = pd.DataFrame()
    summary_f1_macros_test = pd.DataFrame()
    summary_f1_macros_full = pd.DataFrame()
    
    for i in range(2):
        X = data_sets[i][0]
        y = data_sets[i][1]
        X_full = data_sets[i][2]
        y_full = data_sets[i][3]
    
        X_train_og, X_test_og, y_train_og, y_test_og = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        
        # Scale the data. "sc" stands for scaled.
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_og)
        X_test_sc = scaler.transform(X_test_og)
        X_full_sc = scaler.transform(X_full)
        
        # Ensure same k-fold splits across same seed
        kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=False)
        
        # Create ADASYN pipeline
        ad = ADASYN(sampling_strategy = 'auto', n_neighbors = 5, random_state = seed)
        pipeline_setup = [('ad', ad)]
        
        # Logistic Regression
        print("Logistic Regression")
        print(datetime.datetime.now().time())
        
        lr = LogisticRegression(solver = 'saga', random_state = seed)
            
        pipeline_lr = Pipeline(pipeline_setup + [('lr', lr)])
        param_grid_lr = [{'lr__penalty': ['l1', 'l2'],
                          'lr__C': [100, 10, 1, 0.1]}]
        gs_lr = GridSearchCV(estimator = pipeline_lr, param_grid = param_grid_lr, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_lr.fit(X_train_sc, y_train_og)
        lr_pred_train = gs_lr.predict(X_train_sc)
        lr_pred_test = gs_lr.predict(X_test_sc)
        lr_pred_full = gs_lr.predict(X_full_sc)
        lr_f1_train_report = classification_report(y_train_og, lr_pred_train, output_dict=True)
        lr_f1_test_report = classification_report(y_test_og, lr_pred_test, output_dict=True)
        lr_f1_full_report = classification_report(y_full, lr_pred_full, output_dict=True)
        lr_probs_train = gs_lr.predict_proba(X_train_sc)
        lr_probs_test = gs_lr.predict_proba(X_test_sc)
        lr_probs_full = gs_lr.predict_proba(X_full_sc)
        lr_best_params = gs_lr.best_params_
        cv_results[i].append(gs_lr.cv_results_)        
        
        # Quadratic Discriminant Analysis
        print("Quadratic Discriminant Analysis")
        print(datetime.datetime.now().time())
        
        qda = QuadraticDiscriminantAnalysis()
            
        pipeline_qda = Pipeline(pipeline_setup + [('qda', qda)])
        param_grid_qda = [{'qda__reg_param': [1, 0.7, 0.3, 0]}]
        gs_qda = GridSearchCV(estimator = pipeline_qda, param_grid = param_grid_qda, 
                              cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_qda.fit(X_train_sc, y_train_og)
        qda_pred_train = gs_qda.predict(X_train_sc)
        qda_pred_test = gs_qda.predict(X_test_sc)
        qda_pred_full = gs_qda.predict(X_full_sc)
        qda_f1_train_report = classification_report(y_train_og, qda_pred_train, output_dict=True)
        qda_f1_test_report = classification_report(y_test_og, qda_pred_test, output_dict=True)
        qda_f1_full_report = classification_report(y_full, qda_pred_full, output_dict=True)
        qda_probs_train = gs_qda.predict_proba(X_train_sc)
        qda_probs_test = gs_qda.predict_proba(X_test_sc)
        qda_probs_full = gs_qda.predict_proba(X_full_sc)
        qda_best_params = gs_qda.best_params_
        cv_results[i].append(gs_qda.cv_results_)
        
        # Nearest Neighbors
        print("Nearest Neighbors")
        print(datetime.datetime.now().time())
        
        nn = KNeighborsClassifier(algorithm = 'auto', weights = 'distance')
            
        pipeline_nn = Pipeline(pipeline_setup + [('nn', nn)])
        param_grid_nn = [{'nn__n_neighbors': [2, 3, 4, 5],
                          'nn__leaf_size': [2, 3, 4, 30, 60]}]
        gs_nn = GridSearchCV(estimator = pipeline_nn, param_grid = param_grid_nn, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_nn.fit(X_train_sc, y_train_og)
        nn_pred_train = gs_nn.predict(X_train_sc)
        nn_pred_test = gs_nn.predict(X_test_sc)
        nn_pred_full = gs_nn.predict(X_full_sc)
        nn_f1_train_report = classification_report(y_train_og, nn_pred_train, output_dict=True)
        nn_f1_test_report = classification_report(y_test_og, nn_pred_test, output_dict=True)
        nn_f1_full_report = classification_report(y_full, nn_pred_full, output_dict=True)
        nn_probs_train = gs_nn.predict_proba(X_train_sc)
        nn_probs_test = gs_nn.predict_proba(X_test_sc)
        nn_probs_full = gs_nn.predict_proba(X_full_sc)
        nn_best_params = gs_nn.best_params_
        cv_results[i].append(gs_nn.cv_results_)
        
        # Radial Basis Function Support-Vector Machine
        print("Radial Basis Function Support-Vector Machine")
        print(datetime.datetime.now().time())
        
        rbfsvm = SVC(kernel = 'rbf', probability = True, tol = 1e-3, random_state = seed)
            
        pipeline_rbfsvm = Pipeline(pipeline_setup + [('rbfsvm', rbfsvm)])
        param_grid_rbfsvm = [{'rbfsvm__C': [125, 100, 75],
                              'rbfsvm__gamma': [0.5, 0.1, 0.05]}]
        gs_rbfsvm = GridSearchCV(estimator = pipeline_rbfsvm, param_grid = param_grid_rbfsvm, 
                                 cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_rbfsvm.fit(X_train_sc, y_train_og)
        rbfsvm_pred_train = gs_rbfsvm.predict(X_train_sc)
        rbfsvm_pred_test = gs_rbfsvm.predict(X_test_sc)
        rbfsvm_pred_full = gs_rbfsvm.predict(X_full_sc)
        rbfsvm_f1_train_report = classification_report(y_train_og, rbfsvm_pred_train, output_dict=True)
        rbfsvm_f1_test_report = classification_report(y_test_og, rbfsvm_pred_test, output_dict=True)
        rbfsvm_f1_full_report = classification_report(y_full, rbfsvm_pred_full, output_dict=True)
        rbfsvm_probs_train = gs_rbfsvm.predict_proba(X_train_sc)
        rbfsvm_probs_test = gs_rbfsvm.predict_proba(X_test_sc)
        rbfsvm_probs_full = gs_rbfsvm.predict_proba(X_full_sc)
        rbfsvm_best_params = gs_rbfsvm.best_params_
        cv_results[i].append(gs_rbfsvm.cv_results_)
        
        # Naive Bayes
        print("Naive Bayes")
        print(datetime.datetime.now().time())
        
        nb = GaussianNB()
            
        pipeline_nb = Pipeline(pipeline_setup + [('nb', nb)])
        param_grid_nb = [{'nb__var_smoothing': [1e-4, 1e-3, 1e-2]}]
        gs_nb = GridSearchCV(estimator = pipeline_nb, param_grid = param_grid_nb, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_nb.fit(X_train_sc, y_train_og)
        nb_pred_train = gs_nb.predict(X_train_sc)
        nb_pred_test = gs_nb.predict(X_test_sc)
        nb_pred_full = gs_nb.predict(X_full_sc)
        nb_f1_train_report = classification_report(y_train_og, nb_pred_train, output_dict=True)
        nb_f1_test_report = classification_report(y_test_og, nb_pred_test, output_dict=True)
        nb_f1_full_report = classification_report(y_full, nb_pred_full, output_dict=True)
        nb_probs_train = gs_nb.predict_proba(X_train_sc)
        nb_probs_test = gs_nb.predict_proba(X_test_sc)
        nb_probs_full = gs_nb.predict_proba(X_full_sc)
        nb_best_params = gs_nb.best_params_
        cv_results[i].append(gs_nb.cv_results_)
        
        # Artificial Neural Network
        print("Artificial Neural Network")
        print(datetime.datetime.now().time())
        
        ann = MLPClassifier(random_state = seed, max_iter = 500, solver = 'adam',
                            activation = 'relu')
            
        pipeline_ann = Pipeline(pipeline_setup + [('ann', ann)])
        param_grid_ann = [{'ann__hidden_layer_sizes': [(20), (22), (20, 20), (22, 22)],
                   'ann__alpha': [0.007, 0.003, 0.001]}]
        gs_ann = GridSearchCV(estimator = pipeline_ann, param_grid = param_grid_ann, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_ann.fit(X_train_sc, y_train_og)
        ann_pred_train = gs_ann.predict(X_train_sc)
        ann_pred_test = gs_ann.predict(X_test_sc)
        ann_pred_full = gs_ann.predict(X_full_sc)
        ann_f1_train_report = classification_report(y_train_og, ann_pred_train, output_dict=True)
        ann_f1_test_report = classification_report(y_test_og, ann_pred_test, output_dict=True)
        ann_f1_full_report = classification_report(y_full, ann_pred_full, output_dict=True)
        ann_probs_train = gs_ann.predict_proba(X_train_sc)
        ann_probs_test = gs_ann.predict_proba(X_test_sc)
        ann_probs_full = gs_ann.predict_proba(X_full_sc)
        ann_best_params = gs_ann.best_params_
        cv_results[i].append(gs_ann.cv_results_)
        
        # Random Forest
        print("Random Forest")
        print(datetime.datetime.now().time())
        
        rf = RandomForestClassifier(random_state = seed, criterion = 'entropy',
                                    n_estimators = 500, min_samples_split = 2)
            
        pipeline_rf = Pipeline(pipeline_setup + [('rf', rf)])
        param_grid_rf = [{'rf__max_depth': [20, 30],
                          'rf__min_samples_leaf': [1, 2]}]
        gs_rf = GridSearchCV(estimator = pipeline_rf, param_grid = param_grid_rf, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_rf.fit(X_train_sc, y_train_og)
        rf_pred_train = gs_rf.predict(X_train_sc)
        rf_pred_test = gs_rf.predict(X_test_sc)
        rf_pred_full = gs_rf.predict(X_full_sc)
        rf_f1_train_report = classification_report(y_train_og, rf_pred_train, output_dict=True)
        rf_f1_test_report = classification_report(y_test_og, rf_pred_test, output_dict=True)
        rf_f1_full_report = classification_report(y_full, rf_pred_full, output_dict=True)
        rf_probs_train = gs_rf.predict_proba(X_train_sc)
        rf_probs_test = gs_rf.predict_proba(X_test_sc)
        rf_probs_full = gs_rf.predict_proba(X_full_sc)
        rf_best_params = gs_rf.best_params_
        cv_results[i].append(gs_rf.cv_results_)
        
        # AdaBoost Random Forest
        print("AdaBoost Random Forest")
        print(datetime.datetime.now().time())
        
        ab = AdaBoostClassifier(random_state = seed, n_estimators = 500)

        pipeline_ab = Pipeline(pipeline_setup + [('ab', ab)])
        param_grid_ab = [{'ab__learning_rate': [0.1, 0.03, 0.01]}]
        gs_ab = GridSearchCV(estimator = pipeline_ab, param_grid = param_grid_ab, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_ab.fit(X_train_sc, y_train_og)
        ab_pred_train = gs_ab.predict(X_train_sc)
        ab_pred_test = gs_ab.predict(X_test_sc)
        ab_pred_full = gs_ab.predict(X_full_sc)
        ab_f1_train_report = classification_report(y_train_og, ab_pred_train, output_dict=True)
        ab_f1_test_report = classification_report(y_test_og, ab_pred_test, output_dict=True)
        ab_f1_full_report = classification_report(y_full, ab_pred_full, output_dict=True)
        ab_probs_train = gs_ab.predict_proba(X_train_sc)
        ab_probs_test = gs_ab.predict_proba(X_test_sc)
        ab_probs_full = gs_ab.predict_proba(X_full_sc)
        ab_best_params = gs_ab.best_params_
        cv_results[i].append(gs_ab.cv_results_)
        
        # Gradient Boosting Random Forest
        print("Gradient Boosting Random Forest")
        print(datetime.datetime.now().time())
        
        gb = GradientBoostingClassifier(random_state = seed, n_estimators = 500,
                                        min_samples_split = 2)
            
        pipeline_gb = Pipeline(pipeline_setup + [('gb', gb)])
        param_grid_gb = [{'gb__max_depth': [4, 5],
                          'gb__min_samples_leaf': [4, 5, 6],
                          'gb__learning_rate': [0.25, 0.33, 0.5]}]
        gs_gb = GridSearchCV(estimator = pipeline_gb, param_grid = param_grid_gb, 
                             cv = kf, n_jobs = -1, verbose = 5, scoring = 'f1_macro')
        gs_gb.fit(X_train_sc, y_train_og)
        gb_pred_train = gs_gb.predict(X_train_sc)
        gb_pred_test = gs_gb.predict(X_test_sc)
        gb_pred_full = gs_gb.predict(X_full_sc)
        gb_f1_train_report = classification_report(y_train_og, gb_pred_train, output_dict=True)
        gb_f1_test_report = classification_report(y_test_og, gb_pred_test, output_dict=True)
        gb_f1_full_report = classification_report(y_full, gb_pred_full, output_dict=True)
        gb_probs_train = gs_gb.predict_proba(X_train_sc)
        gb_probs_test = gs_gb.predict_proba(X_test_sc)
        gb_probs_full = gs_gb.predict_proba(X_full_sc)
        gb_best_params = gs_gb.best_params_
        cv_results[i].append(gs_gb.cv_results_)
        

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
        
        lr_dict_test = pd.DataFrame.from_dict(lr_f1_test_report)
        qda_dict_test = pd.DataFrame.from_dict(qda_f1_test_report)
        nn_dict_test = pd.DataFrame.from_dict(nn_f1_test_report)
        rbfsvm_dict_test = pd.DataFrame.from_dict(rbfsvm_f1_test_report)
        nb_dict_test = pd.DataFrame.from_dict(nb_f1_test_report)
        ann_dict_test = pd.DataFrame.from_dict(ann_f1_test_report)
        rf_dict_test = pd.DataFrame.from_dict(rf_f1_test_report)
        ab_dict_test = pd.DataFrame.from_dict(ab_f1_test_report)
        gb_dict_test = pd.DataFrame.from_dict(gb_f1_test_report)
        
        lr_dict_full = pd.DataFrame.from_dict(lr_f1_full_report)
        qda_dict_full = pd.DataFrame.from_dict(qda_f1_full_report)
        nn_dict_full = pd.DataFrame.from_dict(nn_f1_full_report)
        rbfsvm_dict_full = pd.DataFrame.from_dict(rbfsvm_f1_full_report)
        nb_dict_full = pd.DataFrame.from_dict(nb_f1_full_report)
        ann_dict_full = pd.DataFrame.from_dict(ann_f1_full_report)
        rf_dict_full = pd.DataFrame.from_dict(rf_f1_full_report)
        ab_dict_full = pd.DataFrame.from_dict(ab_f1_full_report)
        gb_dict_full = pd.DataFrame.from_dict(gb_f1_full_report)
        
        f1_matrix_train = pd.concat([lr_dict_train.transpose()['f1-score'], qda_dict_train.transpose()['f1-score'],
                       nn_dict_train.transpose()['f1-score'], rbfsvm_dict_train.transpose()['f1-score'], 
                       nb_dict_train.transpose()['f1-score'], ann_dict_train.transpose()['f1-score'], 
                       rf_dict_train.transpose()['f1-score'], ab_dict_train.transpose()['f1-score'], 
                       gb_dict_train.transpose()['f1-score']], axis=1)
    
        f1_matrix_test = pd.concat([lr_dict_test.transpose()['f1-score'], qda_dict_test.transpose()['f1-score'],
                       nn_dict_test.transpose()['f1-score'], rbfsvm_dict_test.transpose()['f1-score'], 
                       nb_dict_test.transpose()['f1-score'], ann_dict_test.transpose()['f1-score'], 
                       rf_dict_test.transpose()['f1-score'], ab_dict_test.transpose()['f1-score'], 
                       gb_dict_test.transpose()['f1-score']], axis=1)
    
        f1_matrix_full = pd.concat([lr_dict_full.transpose()['f1-score'], qda_dict_full.transpose()['f1-score'],
                       nn_dict_full.transpose()['f1-score'], rbfsvm_dict_full.transpose()['f1-score'], 
                       nb_dict_full.transpose()['f1-score'], ann_dict_full.transpose()['f1-score'], 
                       rf_dict_full.transpose()['f1-score'], ab_dict_full.transpose()['f1-score'], 
                       gb_dict_full.transpose()['f1-score']], axis=1)
    
        summary_f1_macros_train = pd.concat([summary_f1_macros_train, f1_matrix_train.iloc[[6]]], axis = 0)
        summary_f1_macros_test = pd.concat([summary_f1_macros_test, f1_matrix_test.iloc[[6]]], axis = 0)
        summary_f1_macros_full = pd.concat([summary_f1_macros_full, f1_matrix_full.iloc[[6]]], axis = 0)
    
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
        
        dfcv.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2dfcvdata'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        f1_matrix_train.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2f1traindata'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        f1_matrix_test.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2f1testdata'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        f1_matrix_full.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2f1fulldata'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
        dfbp.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2dfbpdata'+str(i)+'seed'+str(seed)+'.csv', index=None, header = True)
    
    summary_f1_macros_train.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2summarytrainseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_macros_test.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2summarytestseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_macros_full.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineTopographyResults\v2summaryfullseed'+str(seed)+'.csv', index=None, header = True)
