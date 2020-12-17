# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:06:14 2020

@author: Timothy

Trying out various MCS architechtures.
Sum just adds probabilities and picks higher.
A: Apply an LR on the train set (imbalanced).
B: Apply an SMOTE and then LR to the train set to balance it.
C: Apply LR on scaled train set.
D: Scale first, then apply SMOTE.
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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline
import datetime

from sameClassOrderNine import sameClassOrderNine
from mcsprob import mcsprob

seeds = [840, 841, 842, 843, 844, 845, 846, 847, 848, 849]

for seed in seeds:
    # Load data, group adequately, and scale
    data = pd.read_csv("/Users/Timothy/Dropbox/Undergrad Thesis Backups/Programming Files Dropbox/dataMasterNoSimpson.csv")
    data_full = pd.read_csv("/Users/Timothy/Dropbox/Undergrad Thesis Backups/Programming Files Dropbox/dataMasterCleanedHasTestNoSimpson.csv")

    X = np.array(data.drop(['Easting', 'Northing', 'GeoUnit', 'Al', 'K', 'Na', 'Sr'], axis=1))
    y = np.array(data['GeoUnit'])
    X_full = np.array(data_full.drop(['Easting', 'Northing', 'GeoUnit', 'Al', 'K', 'Na', 'Sr'], axis=1))
    y_full = np.array(data_full['GeoUnit'])
    
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
        
    cv_results = []
    best_params = []
    summary_f1_macros_train = pd.DataFrame()
    summary_f1_macros_test = pd.DataFrame()
    summary_f1_macros_full = pd.DataFrame()
    
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
    cv_results.append(gs_lr.cv_results_)        
    
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
    cv_results.append(gs_qda.cv_results_)
    
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
    cv_results.append(gs_nn.cv_results_)
    
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
    cv_results.append(gs_rbfsvm.cv_results_)
    
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
    cv_results.append(gs_nb.cv_results_)
    
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
    cv_results.append(gs_ann.cv_results_)
    
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
    cv_results.append(gs_rf.cv_results_)
    
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
    cv_results.append(gs_ab.cv_results_)
    
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
    cv_results.append(gs_gb.cv_results_)
    
    # Create MCS models
    # MCS models
    # Double check to make sure all class orders are the same before creating the multi classifier system (MCS)
    matchingClassOrder = sameClassOrderNine(gs_lr, gs_qda, gs_nn, gs_rbfsvm, gs_nb, gs_ann, gs_rf, gs_ab, gs_gb)
    
    # Create the multi classifier systems (MCS)
    
    classOrder = gs_lr.classes_
    if matchingClassOrder:
        # MCS with top 3, rbfsvm, ann, gb (>70%) with no repetition of decision trees
        mcs3predtrain = mcsprob(classOrder, rbfsvm_probs_train, ann_probs_train, gb_probs_train)
        mcs3train_report = classification_report(y_train_og, mcs3predtrain, output_dict=True)
        # MCS with top 4, rbfsvm, ann, rf, gb (>70%)
        mcs4predtrain = mcsprob(classOrder, rbfsvm_probs_train, ann_probs_train, rf_probs_train, gb_probs_train)
        mcs4train_report = classification_report(y_train_og, mcs4predtrain, output_dict=True)
        # MCS with 5: lr, nn, rbfsvm, ann, gb (>50%) with no repetition of decision trees
        mcs5predtrain = mcsprob(classOrder, lr_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, gb_probs_train)
        mcs5train_report = classification_report(y_train_og, mcs5predtrain, output_dict=True)
        # MCS with top 6, lr, nn, rbfsvm, ann, rf, gb (>50%) with no repetition of decision trees
        mcs6predtrain = mcsprob(classOrder, lr_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, gb_probs_train)
        mcs6train_report = classification_report(y_train_og, mcs6predtrain, output_dict=True)
        # MCS with 8, lr, qda, nn, rbfsvm, ann, rf, ab, gb (>40%)
        mcs8predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs8train_report = classification_report(y_train_og, mcs8predtrain, output_dict=True)
        # MCS with 9, all models, lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb
        mcs9predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, nb_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs9train_report = classification_report(y_train_og, mcs9predtrain, output_dict=True)
        
        # MCS with top 3, rbfsvm, ann, gb (>70%) with no repetition of decision trees
        mcs3predtest = mcsprob(classOrder, rbfsvm_probs_test, ann_probs_test, gb_probs_test)
        mcs3test_report = classification_report(y_test_og, mcs3predtest, output_dict=True)
        # MCS with top 4, rbfsvm, ann, rf, gb (>70%)
        mcs4predtest = mcsprob(classOrder, rbfsvm_probs_test, ann_probs_test, rf_probs_test, gb_probs_test)
        mcs4test_report = classification_report(y_test_og, mcs4predtest, output_dict=True)
        # MCS with 5: lr, nn, rbfsvm, ann, gb (>50%) with no repetition of decision trees
        mcs5predtest = mcsprob(classOrder, lr_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, gb_probs_test)
        mcs5test_report = classification_report(y_test_og, mcs5predtest, output_dict=True)
        # MCS with top 6, lr, nn, rbfsvm, ann, rf, gb (>50%) with no repetition of decision trees
        mcs6predtest = mcsprob(classOrder, lr_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, gb_probs_test)
        mcs6test_report = classification_report(y_test_og, mcs6predtest, output_dict=True)
        # MCS with 8, lr, qda, nn, rbfsvm, ann, rf, ab, gb (>40%)
        mcs8predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs8test_report = classification_report(y_test_og, mcs8predtest, output_dict=True)
        # MCS with 9, all models, lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb
        mcs9predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, nb_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs9test_report = classification_report(y_test_og, mcs9predtest, output_dict=True)
        
        # MCS with top 3, rbfsvm, ann, gb (>70%) with no repetition of decision trees
        mcs3predfull = mcsprob(classOrder, rbfsvm_probs_full, ann_probs_full, gb_probs_full)
        mcs3full_report = classification_report(y_full, mcs3predfull, output_dict=True)
        # MCS with top 4, rbfsvm, ann, rf, gb (>70%)
        mcs4predfull = mcsprob(classOrder, rbfsvm_probs_full, ann_probs_full, rf_probs_full, gb_probs_full)
        mcs4full_report = classification_report(y_full, mcs4predfull, output_dict=True)
        # MCS with 5: lr, nn, rbfsvm, ann, gb (>50%) with no repetition of decision trees
        mcs5predfull = mcsprob(classOrder, lr_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, gb_probs_full)
        mcs5full_report = classification_report(y_full, mcs5predfull, output_dict=True)
        # MCS with top 6, lr, nn, rbfsvm, ann, rf, gb (>50%) with no repetition of decision trees
        mcs6predfull = mcsprob(classOrder, lr_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, gb_probs_full)
        mcs6full_report = classification_report(y_full, mcs6predfull, output_dict=True)
        # MCS with 8, lr, qda, nn, rbfsvm, ann, rf, ab, gb (>40%)
        mcs8predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs8full_report = classification_report(y_full, mcs8predfull, output_dict=True)
        # MCS with 9, all models, lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb
        mcs9predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, nb_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs9full_report = classification_report(y_full, mcs9predfull, output_dict=True)
        
    else:
        print('Failed matching class order')
    
    # AMCS: Applies LR to the imbalanced train set
    # LR 3 uses RBFSVM, ANN, and GB
    lr3_train = np.concatenate((rbfsvm_probs_train, ann_probs_train, gb_probs_train), axis=1)
    lr3_test = np.concatenate((rbfsvm_probs_test, ann_probs_test, gb_probs_test), axis=1)
    lr3_full = np.concatenate((rbfsvm_probs_full, ann_probs_full, gb_probs_full), axis=1)

    alr3_mcs = LogisticRegression(random_state = seed)
    alr3_mcs.fit(lr3_train, y_train_og)
    alr3_mcs_pred_train = alr3_mcs.predict(lr3_train)
    alr3_mcs_f1_report_train = classification_report(y_train_og, alr3_mcs_pred_train, output_dict=True)
    alr3_mcs_pred_test = alr3_mcs.predict(lr3_test)
    alr3_mcs_f1_report_test = classification_report(y_test_og, alr3_mcs_pred_test, output_dict=True)
    alr3_mcs_pred_full = alr3_mcs.predict(lr3_full)
    alr3_mcs_f1_report_full = classification_report(y_full, alr3_mcs_pred_full, output_dict=True)
    
    # LR 4 uses RBFSVM, ANN, RF, and GB
    lr4_train = np.concatenate((rbfsvm_probs_train, ann_probs_train, rf_probs_train, gb_probs_train), axis=1)
    lr4_test = np.concatenate((rbfsvm_probs_test, ann_probs_test, rf_probs_test, gb_probs_test), axis=1)
    lr4_full = np.concatenate((rbfsvm_probs_full, ann_probs_full, rf_probs_full, gb_probs_full), axis=1)

    alr4_mcs = LogisticRegression(random_state = seed)
    alr4_mcs.fit(lr4_train, y_train_og)
    alr4_mcs_pred_train = alr4_mcs.predict(lr4_train)
    alr4_mcs_f1_report_train = classification_report(y_train_og, alr4_mcs_pred_train, output_dict=True)
    alr4_mcs_pred_test = alr4_mcs.predict(lr4_test)
    alr4_mcs_f1_report_test = classification_report(y_test_og, alr4_mcs_pred_test, output_dict=True)
    alr4_mcs_pred_full = alr4_mcs.predict(lr4_full)
    alr4_mcs_f1_report_full = classification_report(y_full, alr4_mcs_pred_full, output_dict=True)

    # LR 5 uses LR, NN, RBFSVM, ANN, and GB
    lr5_train = np.concatenate((lr_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, gb_probs_train), axis=1)
    lr5_test = np.concatenate((lr_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, gb_probs_test), axis=1)
    lr5_full = np.concatenate((lr_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, gb_probs_full), axis=1)

    alr5_mcs = LogisticRegression(random_state = seed)
    alr5_mcs.fit(lr5_train, y_train_og)
    alr5_mcs_pred_train = alr5_mcs.predict(lr5_train)
    alr5_mcs_f1_report_train = classification_report(y_train_og, alr5_mcs_pred_train, output_dict=True)
    alr5_mcs_pred_test = alr5_mcs.predict(lr5_test)
    alr5_mcs_f1_report_test = classification_report(y_test_og, alr5_mcs_pred_test, output_dict=True)
    alr5_mcs_pred_full = alr5_mcs.predict(lr5_full)
    alr5_mcs_f1_report_full = classification_report(y_full, alr5_mcs_pred_full, output_dict=True)
    
    # LR 6 uses LR, NN, RBFSVM, ANN, RF, and GB
    lr6_train = np.concatenate((lr_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, gb_probs_train), axis=1)
    lr6_test = np.concatenate((lr_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, gb_probs_test), axis=1)
    lr6_full = np.concatenate((lr_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, gb_probs_full), axis=1)

    alr6_mcs = LogisticRegression(random_state = seed)
    alr6_mcs.fit(lr6_train, y_train_og)
    alr6_mcs_pred_train = alr6_mcs.predict(lr6_train)
    alr6_mcs_f1_report_train = classification_report(y_train_og, alr6_mcs_pred_train, output_dict=True)
    alr6_mcs_pred_test = alr6_mcs.predict(lr6_test)
    alr6_mcs_f1_report_test = classification_report(y_test_og, alr6_mcs_pred_test, output_dict=True)
    alr6_mcs_pred_full = alr6_mcs.predict(lr6_full)
    alr6_mcs_f1_report_full = classification_report(y_full, alr6_mcs_pred_full, output_dict=True)

    # LR 8 uses LR, QDA, NN, RBFSVM, ANN, RF, AB, and GB
    lr8_train = np.concatenate((lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr8_test = np.concatenate((lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr8_full = np.concatenate((lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr8_mcs = LogisticRegression(random_state = seed)
    alr8_mcs.fit(lr8_train, y_train_og)
    alr8_mcs_pred_train = alr8_mcs.predict(lr8_train)
    alr8_mcs_f1_report_train = classification_report(y_train_og, alr8_mcs_pred_train, output_dict=True)
    alr8_mcs_pred_test = alr8_mcs.predict(lr8_test)
    alr8_mcs_f1_report_test = classification_report(y_test_og, alr8_mcs_pred_test, output_dict=True)
    alr8_mcs_pred_full = alr8_mcs.predict(lr8_full)
    alr8_mcs_f1_report_full = classification_report(y_full, alr8_mcs_pred_full, output_dict=True)

    # LR 9 uses LR, QDA, NN, RBFSVM, NB, ANN, RF, AB, and GB
    lr9_train = np.concatenate((lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, nb_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr9_test = np.concatenate((lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, nb_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr9_full = np.concatenate((lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, nb_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr9_mcs = LogisticRegression(random_state = seed)
    alr9_mcs.fit(lr9_train, y_train_og)
    alr9_mcs_pred_train = alr9_mcs.predict(lr9_train)
    alr9_mcs_f1_report_train = classification_report(y_train_og, alr9_mcs_pred_train, output_dict=True)
    alr9_mcs_pred_test = alr9_mcs.predict(lr9_test)
    alr9_mcs_f1_report_test = classification_report(y_test_og, alr9_mcs_pred_test, output_dict=True)
    alr9_mcs_pred_full = alr9_mcs.predict(lr9_full)
    alr9_mcs_f1_report_full = classification_report(y_full, alr9_mcs_pred_full, output_dict=True)


    # BMCS: Applies LR to a SMOTE-ed data set of train set
    # LR 3 uses RBFSVM, ANN, and GB
    sm3 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b3, y_train_b3 = sm3.fit_resample(lr3_train, y_train_og)
    
    blr3_mcs = LogisticRegression(random_state = seed)
    blr3_mcs.fit(X_train_b3, y_train_b3)
    blr3_mcs_pred_train = blr3_mcs.predict(lr3_train)
    blr3_mcs_f1_report_train = classification_report(y_train_og, blr3_mcs_pred_train, output_dict=True)
    blr3_mcs_pred_test = blr3_mcs.predict(lr3_test)
    blr3_mcs_f1_report_test = classification_report(y_test_og, blr3_mcs_pred_test, output_dict=True)
    blr3_mcs_pred_full = blr3_mcs.predict(lr3_full)
    blr3_mcs_f1_report_full = classification_report(y_full, blr3_mcs_pred_full, output_dict=True)
    
    # LR 4 uses RBFSVM, ANN, RF, and GB
    sm4 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b4, y_train_b4 = sm4.fit_resample(lr4_train, y_train_og)
    
    blr4_mcs = LogisticRegression(random_state = seed)
    blr4_mcs.fit(X_train_b4, y_train_b4)
    blr4_mcs_pred_train = blr4_mcs.predict(lr4_train)
    blr4_mcs_f1_report_train = classification_report(y_train_og, blr4_mcs_pred_train, output_dict=True)
    blr4_mcs_pred_test = blr4_mcs.predict(lr4_test)
    blr4_mcs_f1_report_test = classification_report(y_test_og, blr4_mcs_pred_test, output_dict=True)
    blr4_mcs_pred_full = blr4_mcs.predict(lr4_full)
    blr4_mcs_f1_report_full = classification_report(y_full, blr4_mcs_pred_full, output_dict=True)
    
    # LR 5 uses LR, NN, RBFSVM, ANN, and GB
    sm5 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b5, y_train_b5 = sm5.fit_resample(lr5_train, y_train_og)
    
    blr5_mcs = LogisticRegression(random_state = seed)
    blr5_mcs.fit(X_train_b5, y_train_b5)
    blr5_mcs_pred_train = blr5_mcs.predict(lr5_train)
    blr5_mcs_f1_report_train = classification_report(y_train_og, blr5_mcs_pred_train, output_dict=True)
    blr5_mcs_pred_test = blr5_mcs.predict(lr5_test)
    blr5_mcs_f1_report_test = classification_report(y_test_og, blr5_mcs_pred_test, output_dict=True)
    blr5_mcs_pred_full = blr5_mcs.predict(lr5_full)
    blr5_mcs_f1_report_full = classification_report(y_full, blr5_mcs_pred_full, output_dict=True)
    
    # LR 6 uses LR, NN, RBFSVM, ANN, RF, and GB
    sm6 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b6, y_train_b6 = sm6.fit_resample(lr6_train, y_train_og)
    
    blr6_mcs = LogisticRegression(random_state = seed)
    blr6_mcs.fit(X_train_b6, y_train_b6)
    blr6_mcs_pred_train = blr6_mcs.predict(lr6_train)
    blr6_mcs_f1_report_train = classification_report(y_train_og, blr6_mcs_pred_train, output_dict=True)
    blr6_mcs_pred_test = blr6_mcs.predict(lr6_test)
    blr6_mcs_f1_report_test = classification_report(y_test_og, blr6_mcs_pred_test, output_dict=True)
    blr6_mcs_pred_full = blr6_mcs.predict(lr6_full)
    blr6_mcs_f1_report_full = classification_report(y_full, blr6_mcs_pred_full, output_dict=True)
    
    # LR 8 uses LR, QDA, NN, RBFSVM, ANN, RF, AB, and GB
    sm8 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b8, y_train_b8 = sm8.fit_resample(lr8_train, y_train_og)
    
    blr8_mcs = LogisticRegression(random_state = seed)
    blr8_mcs.fit(X_train_b8, y_train_b8)
    blr8_mcs_pred_train = blr8_mcs.predict(lr8_train)
    blr8_mcs_f1_report_train = classification_report(y_train_og, blr8_mcs_pred_train, output_dict=True)
    blr8_mcs_pred_test = blr8_mcs.predict(lr8_test)
    blr8_mcs_f1_report_test = classification_report(y_test_og, blr8_mcs_pred_test, output_dict=True)
    blr8_mcs_pred_full = blr8_mcs.predict(lr8_full)
    blr8_mcs_f1_report_full = classification_report(y_full, blr8_mcs_pred_full, output_dict=True)
    
    # LR 9 uses LR, QDA, NN, RBFSVM, NB, ANN, RF, AB, and GB
    sm9 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b9, y_train_b9 = sm9.fit_resample(lr9_train, y_train_og)
    
    blr9_mcs = LogisticRegression(random_state = seed)
    blr9_mcs.fit(X_train_b9, y_train_b9)
    blr9_mcs_pred_train = blr9_mcs.predict(lr9_train)
    blr9_mcs_f1_report_train = classification_report(y_train_og, blr9_mcs_pred_train, output_dict=True)
    blr9_mcs_pred_test = blr9_mcs.predict(lr9_test)
    blr9_mcs_f1_report_test = classification_report(y_test_og, blr9_mcs_pred_test, output_dict=True)
    blr9_mcs_pred_full = blr9_mcs.predict(lr9_full)
    blr9_mcs_f1_report_full = classification_report(y_full, blr9_mcs_pred_full, output_dict=True)


    # CMCS: Applies LR to a scaled data set of train set
    # LR 3 uses RBFSVM, ANN, and GB
    scal3 = StandardScaler()
    X_train_c3 = scaler.fit_transform(lr3_train)
    X_test_c3 = scaler.transform(lr3_test)
    X_full_c3 = scaler.transform(lr3_full)
    
    clr3_mcs = LogisticRegression(random_state = seed)
    clr3_mcs.fit(X_train_c3, y_train_og)
    clr3_mcs_pred_train = clr3_mcs.predict(X_train_c3)
    clr3_mcs_f1_report_train = classification_report(y_train_og, clr3_mcs_pred_train, output_dict=True)
    clr3_mcs_pred_test = clr3_mcs.predict(X_test_c3)
    clr3_mcs_f1_report_test = classification_report(y_test_og, clr3_mcs_pred_test, output_dict=True)
    clr3_mcs_pred_full = clr3_mcs.predict(X_full_c3)
    clr3_mcs_f1_report_full = classification_report(y_full, clr3_mcs_pred_full, output_dict=True)
    
    # LR 4 uses RBFSVM, ANN, RF, and GB
    scal4 = StandardScaler()
    X_train_c4 = scaler.fit_transform(lr4_train)
    X_test_c4 = scaler.transform(lr4_test)
    X_full_c4 = scaler.transform(lr4_full)
    
    clr4_mcs = LogisticRegression(random_state = seed)
    clr4_mcs.fit(X_train_c4, y_train_og)
    clr4_mcs_pred_train = clr4_mcs.predict(X_train_c4)
    clr4_mcs_f1_report_train = classification_report(y_train_og, clr4_mcs_pred_train, output_dict=True)
    clr4_mcs_pred_test = clr4_mcs.predict(X_test_c4)
    clr4_mcs_f1_report_test = classification_report(y_test_og, clr4_mcs_pred_test, output_dict=True)
    clr4_mcs_pred_full = clr4_mcs.predict(X_full_c4)
    clr4_mcs_f1_report_full = classification_report(y_full, clr4_mcs_pred_full, output_dict=True)
    
    # LR 5 uses LR, NN, RBFSVM, ANN, and GB
    scal5 = StandardScaler()
    X_train_c5 = scaler.fit_transform(lr5_train)
    X_test_c5 = scaler.transform(lr5_test)
    X_full_c5 = scaler.transform(lr5_full)
    
    clr5_mcs = LogisticRegression(random_state = seed)
    clr5_mcs.fit(X_train_c5, y_train_og)
    clr5_mcs_pred_train = clr5_mcs.predict(X_train_c5)
    clr5_mcs_f1_report_train = classification_report(y_train_og, clr5_mcs_pred_train, output_dict=True)
    clr5_mcs_pred_test = clr5_mcs.predict(X_test_c5)
    clr5_mcs_f1_report_test = classification_report(y_test_og, clr5_mcs_pred_test, output_dict=True)
    clr5_mcs_pred_full = clr5_mcs.predict(X_full_c5)
    clr5_mcs_f1_report_full = classification_report(y_full, clr5_mcs_pred_full, output_dict=True)
    
    # LR 6 uses LR, NN, RBFSVM, ANN, RF, and GB
    scal6 = StandardScaler()
    X_train_c6 = scaler.fit_transform(lr6_train)
    X_test_c6 = scaler.transform(lr6_test)
    X_full_c6 = scaler.transform(lr6_full)
    
    clr6_mcs = LogisticRegression(random_state = seed)
    clr6_mcs.fit(X_train_c6, y_train_og)
    clr6_mcs_pred_train = clr6_mcs.predict(X_train_c6)
    clr6_mcs_f1_report_train = classification_report(y_train_og, clr6_mcs_pred_train, output_dict=True)
    clr6_mcs_pred_test = clr6_mcs.predict(X_test_c6)
    clr6_mcs_f1_report_test = classification_report(y_test_og, clr6_mcs_pred_test, output_dict=True)
    clr6_mcs_pred_full = clr6_mcs.predict(X_full_c6)
    clr6_mcs_f1_report_full = classification_report(y_full, clr6_mcs_pred_full, output_dict=True)
    
    # LR 8 uses LR, QDA, NN, RBFSVM, ANN, RF, AB, and GB
    scal8 = StandardScaler()
    X_train_c8 = scaler.fit_transform(lr8_train)
    X_test_c8 = scaler.transform(lr8_test)
    X_full_c8 = scaler.transform(lr8_full)
    
    clr8_mcs = LogisticRegression(random_state = seed)
    clr8_mcs.fit(X_train_c8, y_train_og)
    clr8_mcs_pred_train = clr8_mcs.predict(X_train_c8)
    clr8_mcs_f1_report_train = classification_report(y_train_og, clr8_mcs_pred_train, output_dict=True)
    clr8_mcs_pred_test = clr8_mcs.predict(X_test_c8)
    clr8_mcs_f1_report_test = classification_report(y_test_og, clr8_mcs_pred_test, output_dict=True)
    clr8_mcs_pred_full = clr8_mcs.predict(X_full_c8)
    clr8_mcs_f1_report_full = classification_report(y_full, clr8_mcs_pred_full, output_dict=True)
    
    # LR 9 uses LR, QDA, NN, RBFSVM, NB, ANN, RF, AB, and GB
    scal9 = StandardScaler()
    X_train_c9 = scaler.fit_transform(lr9_train)
    X_test_c9 = scaler.transform(lr9_test)
    X_full_c9 = scaler.transform(lr9_full)
    
    clr9_mcs = LogisticRegression(random_state = seed)
    clr9_mcs.fit(X_train_c9, y_train_og)
    clr9_mcs_pred_train = clr9_mcs.predict(X_train_c9)
    clr9_mcs_f1_report_train = classification_report(y_train_og, clr9_mcs_pred_train, output_dict=True)
    clr9_mcs_pred_test = clr9_mcs.predict(X_test_c9)
    clr9_mcs_f1_report_test = classification_report(y_test_og, clr9_mcs_pred_test, output_dict=True)
    clr9_mcs_pred_full = clr9_mcs.predict(X_full_c9)
    clr9_mcs_f1_report_full = classification_report(y_full, clr9_mcs_pred_full, output_dict=True)
    
    
    # DMCS: Applies LR to a SMOTE-ed Scaled data set of train set
    # LR 3 uses RBFSVM, ANN, and GB
    smd3 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d3, y_train_d3 = smd3.fit_resample(X_train_c3, y_train_og)
    
    dlr3_mcs = LogisticRegression(random_state = seed)
    dlr3_mcs.fit(X_train_d3, y_train_d3)
    dlr3_mcs_pred_train = dlr3_mcs.predict(X_train_c3)
    dlr3_mcs_f1_report_train = classification_report(y_train_og, dlr3_mcs_pred_train, output_dict=True)
    dlr3_mcs_pred_test = dlr3_mcs.predict(X_test_c3)
    dlr3_mcs_f1_report_test = classification_report(y_test_og, dlr3_mcs_pred_test, output_dict=True)
    dlr3_mcs_pred_full = dlr3_mcs.predict(X_full_c3)
    dlr3_mcs_f1_report_full = classification_report(y_full, dlr3_mcs_pred_full, output_dict=True)
    
    # LR 4 uses RBFSVM, ANN, RF, and GB
    smd4 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d4, y_train_d4 = smd4.fit_resample(X_train_c4, y_train_og)
    
    dlr4_mcs = LogisticRegression(random_state = seed)
    dlr4_mcs.fit(X_train_d4, y_train_d4)
    dlr4_mcs_pred_train = dlr4_mcs.predict(X_train_c4)
    dlr4_mcs_f1_report_train = classification_report(y_train_og, dlr4_mcs_pred_train, output_dict=True)
    dlr4_mcs_pred_test = dlr4_mcs.predict(X_test_c4)
    dlr4_mcs_f1_report_test = classification_report(y_test_og, dlr4_mcs_pred_test, output_dict=True)
    dlr4_mcs_pred_full = dlr4_mcs.predict(X_full_c4)
    dlr4_mcs_f1_report_full = classification_report(y_full, dlr4_mcs_pred_full, output_dict=True)
    
    # LR 5 uses LR, NN, RBFSVM, ANN, and GB
    smd5 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d5, y_train_d5 = smd5.fit_resample(X_train_c5, y_train_og)
    
    dlr5_mcs = LogisticRegression(random_state = seed)
    dlr5_mcs.fit(X_train_d5, y_train_d5)
    dlr5_mcs_pred_train = dlr5_mcs.predict(X_train_c5)
    dlr5_mcs_f1_report_train = classification_report(y_train_og, dlr5_mcs_pred_train, output_dict=True)
    dlr5_mcs_pred_test = dlr5_mcs.predict(X_test_c5)
    dlr5_mcs_f1_report_test = classification_report(y_test_og, dlr5_mcs_pred_test, output_dict=True)
    dlr5_mcs_pred_full = dlr5_mcs.predict(X_full_c5)
    dlr5_mcs_f1_report_full = classification_report(y_full, dlr5_mcs_pred_full, output_dict=True)
    
    # LR 6 uses LR, NN, RBFSVM, ANN, RF, and GB
    smd6 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d6, y_train_d6 = smd6.fit_resample(X_train_c6, y_train_og)
    
    dlr6_mcs = LogisticRegression(random_state = seed)
    dlr6_mcs.fit(X_train_d6, y_train_d6)
    dlr6_mcs_pred_train = dlr6_mcs.predict(X_train_c6)
    dlr6_mcs_f1_report_train = classification_report(y_train_og, dlr6_mcs_pred_train, output_dict=True)
    dlr6_mcs_pred_test = dlr6_mcs.predict(X_test_c6)
    dlr6_mcs_f1_report_test = classification_report(y_test_og, dlr6_mcs_pred_test, output_dict=True)
    dlr6_mcs_pred_full = dlr6_mcs.predict(X_full_c6)
    dlr6_mcs_f1_report_full = classification_report(y_full, dlr6_mcs_pred_full, output_dict=True)
    
    # LR 8 uses LR, QDA, NN, RBFSVM, ANN, RF, AB, and GB
    smd8 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d8, y_train_d8 = smd8.fit_resample(X_train_c8, y_train_og)
    
    dlr8_mcs = LogisticRegression(random_state = seed)
    dlr8_mcs.fit(X_train_d8, y_train_d8)
    dlr8_mcs_pred_train = dlr8_mcs.predict(X_train_c8)
    dlr8_mcs_f1_report_train = classification_report(y_train_og, dlr8_mcs_pred_train, output_dict=True)
    dlr8_mcs_pred_test = dlr8_mcs.predict(X_test_c8)
    dlr8_mcs_f1_report_test = classification_report(y_test_og, dlr8_mcs_pred_test, output_dict=True)
    dlr8_mcs_pred_full = dlr8_mcs.predict(X_full_c8)
    dlr8_mcs_f1_report_full = classification_report(y_full, dlr8_mcs_pred_full, output_dict=True)
    
    # LR 9 uses LR, QDA, NN, RBFSVM, NB, ANN, RF, AB, and GB
    smd9 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d9, y_train_d9 = smd9.fit_resample(X_train_c9, y_train_og)
    
    dlr9_mcs = LogisticRegression(random_state = seed)
    dlr9_mcs.fit(X_train_d9, y_train_d9)
    dlr9_mcs_pred_train = dlr9_mcs.predict(X_train_c9)
    dlr9_mcs_f1_report_train = classification_report(y_train_og, dlr9_mcs_pred_train, output_dict=True)
    dlr9_mcs_pred_test = dlr9_mcs.predict(X_test_c9)
    dlr9_mcs_f1_report_test = classification_report(y_test_og, dlr9_mcs_pred_test, output_dict=True)
    dlr9_mcs_pred_full = dlr9_mcs.predict(X_full_c9)
    dlr9_mcs_f1_report_full = classification_report(y_full, dlr9_mcs_pred_full, output_dict=True)
    
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
    mcs3_dict_train = pd.DataFrame.from_dict(mcs3train_report)
    mcs4_dict_train = pd.DataFrame.from_dict(mcs4train_report)
    mcs5_dict_train = pd.DataFrame.from_dict(mcs5train_report)
    mcs6_dict_train = pd.DataFrame.from_dict(mcs6train_report)
    mcs8_dict_train = pd.DataFrame.from_dict(mcs8train_report)
    mcs9_dict_train = pd.DataFrame.from_dict(mcs9train_report)
    amcs3_dict_train = pd.DataFrame.from_dict(alr3_mcs_f1_report_train)
    amcs4_dict_train = pd.DataFrame.from_dict(alr4_mcs_f1_report_train)
    amcs5_dict_train = pd.DataFrame.from_dict(alr5_mcs_f1_report_train)
    amcs6_dict_train = pd.DataFrame.from_dict(alr6_mcs_f1_report_train)
    amcs8_dict_train = pd.DataFrame.from_dict(alr8_mcs_f1_report_train)
    amcs9_dict_train = pd.DataFrame.from_dict(alr9_mcs_f1_report_train)
    bmcs3_dict_train = pd.DataFrame.from_dict(blr3_mcs_f1_report_train)
    bmcs4_dict_train = pd.DataFrame.from_dict(blr4_mcs_f1_report_train)
    bmcs5_dict_train = pd.DataFrame.from_dict(blr5_mcs_f1_report_train)
    bmcs6_dict_train = pd.DataFrame.from_dict(blr6_mcs_f1_report_train)
    bmcs8_dict_train = pd.DataFrame.from_dict(blr8_mcs_f1_report_train)
    bmcs9_dict_train = pd.DataFrame.from_dict(blr9_mcs_f1_report_train)
    cmcs3_dict_train = pd.DataFrame.from_dict(clr3_mcs_f1_report_train)
    cmcs4_dict_train = pd.DataFrame.from_dict(clr4_mcs_f1_report_train)
    cmcs5_dict_train = pd.DataFrame.from_dict(clr5_mcs_f1_report_train)
    cmcs6_dict_train = pd.DataFrame.from_dict(clr6_mcs_f1_report_train)
    cmcs8_dict_train = pd.DataFrame.from_dict(clr8_mcs_f1_report_train)
    cmcs9_dict_train = pd.DataFrame.from_dict(clr9_mcs_f1_report_train)
    dmcs3_dict_train = pd.DataFrame.from_dict(dlr3_mcs_f1_report_train)
    dmcs4_dict_train = pd.DataFrame.from_dict(dlr4_mcs_f1_report_train)
    dmcs5_dict_train = pd.DataFrame.from_dict(dlr5_mcs_f1_report_train)
    dmcs6_dict_train = pd.DataFrame.from_dict(dlr6_mcs_f1_report_train)
    dmcs8_dict_train = pd.DataFrame.from_dict(dlr8_mcs_f1_report_train)
    dmcs9_dict_train = pd.DataFrame.from_dict(dlr9_mcs_f1_report_train)
    
    
    lr_dict_test = pd.DataFrame.from_dict(lr_f1_test_report)
    qda_dict_test = pd.DataFrame.from_dict(qda_f1_test_report)
    nn_dict_test = pd.DataFrame.from_dict(nn_f1_test_report)
    rbfsvm_dict_test = pd.DataFrame.from_dict(rbfsvm_f1_test_report)
    nb_dict_test = pd.DataFrame.from_dict(nb_f1_test_report)
    ann_dict_test = pd.DataFrame.from_dict(ann_f1_test_report)
    rf_dict_test = pd.DataFrame.from_dict(rf_f1_test_report)
    ab_dict_test = pd.DataFrame.from_dict(ab_f1_test_report)
    gb_dict_test = pd.DataFrame.from_dict(gb_f1_test_report)
    mcs3_dict_test = pd.DataFrame.from_dict(mcs3test_report)
    mcs4_dict_test = pd.DataFrame.from_dict(mcs4test_report)
    mcs5_dict_test = pd.DataFrame.from_dict(mcs5test_report)
    mcs6_dict_test = pd.DataFrame.from_dict(mcs6test_report)
    mcs8_dict_test = pd.DataFrame.from_dict(mcs8test_report)
    mcs9_dict_test = pd.DataFrame.from_dict(mcs9test_report)
    amcs3_dict_test = pd.DataFrame.from_dict(alr3_mcs_f1_report_test)
    amcs4_dict_test = pd.DataFrame.from_dict(alr4_mcs_f1_report_test)
    amcs5_dict_test = pd.DataFrame.from_dict(alr5_mcs_f1_report_test)
    amcs6_dict_test = pd.DataFrame.from_dict(alr6_mcs_f1_report_test)
    amcs8_dict_test = pd.DataFrame.from_dict(alr8_mcs_f1_report_test)
    amcs9_dict_test = pd.DataFrame.from_dict(alr9_mcs_f1_report_test)
    bmcs3_dict_test = pd.DataFrame.from_dict(blr3_mcs_f1_report_test)
    bmcs4_dict_test = pd.DataFrame.from_dict(blr4_mcs_f1_report_test)
    bmcs5_dict_test = pd.DataFrame.from_dict(blr5_mcs_f1_report_test)
    bmcs6_dict_test = pd.DataFrame.from_dict(blr6_mcs_f1_report_test)
    bmcs8_dict_test = pd.DataFrame.from_dict(blr8_mcs_f1_report_test)
    bmcs9_dict_test = pd.DataFrame.from_dict(blr9_mcs_f1_report_test)
    cmcs3_dict_test = pd.DataFrame.from_dict(clr3_mcs_f1_report_test)
    cmcs4_dict_test = pd.DataFrame.from_dict(clr4_mcs_f1_report_test)
    cmcs5_dict_test = pd.DataFrame.from_dict(clr5_mcs_f1_report_test)
    cmcs6_dict_test = pd.DataFrame.from_dict(clr6_mcs_f1_report_test)
    cmcs8_dict_test = pd.DataFrame.from_dict(clr8_mcs_f1_report_test)
    cmcs9_dict_test = pd.DataFrame.from_dict(clr9_mcs_f1_report_test)
    dmcs3_dict_test = pd.DataFrame.from_dict(dlr3_mcs_f1_report_test)
    dmcs4_dict_test = pd.DataFrame.from_dict(dlr4_mcs_f1_report_test)
    dmcs5_dict_test = pd.DataFrame.from_dict(dlr5_mcs_f1_report_test)
    dmcs6_dict_test = pd.DataFrame.from_dict(dlr6_mcs_f1_report_test)
    dmcs8_dict_test = pd.DataFrame.from_dict(dlr8_mcs_f1_report_test)
    dmcs9_dict_test = pd.DataFrame.from_dict(dlr9_mcs_f1_report_test)
    
    lr_dict_full = pd.DataFrame.from_dict(lr_f1_full_report)
    qda_dict_full = pd.DataFrame.from_dict(qda_f1_full_report)
    nn_dict_full = pd.DataFrame.from_dict(nn_f1_full_report)
    rbfsvm_dict_full = pd.DataFrame.from_dict(rbfsvm_f1_full_report)
    nb_dict_full = pd.DataFrame.from_dict(nb_f1_full_report)
    ann_dict_full = pd.DataFrame.from_dict(ann_f1_full_report)
    rf_dict_full = pd.DataFrame.from_dict(rf_f1_full_report)
    ab_dict_full = pd.DataFrame.from_dict(ab_f1_full_report)
    gb_dict_full = pd.DataFrame.from_dict(gb_f1_full_report)
    mcs3_dict_full = pd.DataFrame.from_dict(mcs3full_report)
    mcs4_dict_full = pd.DataFrame.from_dict(mcs4full_report)
    mcs5_dict_full = pd.DataFrame.from_dict(mcs5full_report)
    mcs6_dict_full = pd.DataFrame.from_dict(mcs6full_report)
    mcs8_dict_full = pd.DataFrame.from_dict(mcs8full_report)
    mcs9_dict_full = pd.DataFrame.from_dict(mcs9full_report)
    amcs3_dict_full = pd.DataFrame.from_dict(alr3_mcs_f1_report_full)
    amcs4_dict_full = pd.DataFrame.from_dict(alr4_mcs_f1_report_full)
    amcs5_dict_full = pd.DataFrame.from_dict(alr5_mcs_f1_report_full)
    amcs6_dict_full = pd.DataFrame.from_dict(alr6_mcs_f1_report_full)
    amcs8_dict_full = pd.DataFrame.from_dict(alr8_mcs_f1_report_full)
    amcs9_dict_full = pd.DataFrame.from_dict(alr9_mcs_f1_report_full)
    bmcs3_dict_full = pd.DataFrame.from_dict(blr3_mcs_f1_report_full)
    bmcs4_dict_full = pd.DataFrame.from_dict(blr4_mcs_f1_report_full)
    bmcs5_dict_full = pd.DataFrame.from_dict(blr5_mcs_f1_report_full)
    bmcs6_dict_full = pd.DataFrame.from_dict(blr6_mcs_f1_report_full)
    bmcs8_dict_full = pd.DataFrame.from_dict(blr8_mcs_f1_report_full)
    bmcs9_dict_full = pd.DataFrame.from_dict(blr9_mcs_f1_report_full)
    cmcs3_dict_full = pd.DataFrame.from_dict(clr3_mcs_f1_report_full)
    cmcs4_dict_full = pd.DataFrame.from_dict(clr4_mcs_f1_report_full)
    cmcs5_dict_full = pd.DataFrame.from_dict(clr5_mcs_f1_report_full)
    cmcs6_dict_full = pd.DataFrame.from_dict(clr6_mcs_f1_report_full)
    cmcs8_dict_full = pd.DataFrame.from_dict(clr8_mcs_f1_report_full)
    cmcs9_dict_full = pd.DataFrame.from_dict(clr9_mcs_f1_report_full)
    dmcs3_dict_full = pd.DataFrame.from_dict(dlr3_mcs_f1_report_full)
    dmcs4_dict_full = pd.DataFrame.from_dict(dlr4_mcs_f1_report_full)
    dmcs5_dict_full = pd.DataFrame.from_dict(dlr5_mcs_f1_report_full)
    dmcs6_dict_full = pd.DataFrame.from_dict(dlr6_mcs_f1_report_full)
    dmcs8_dict_full = pd.DataFrame.from_dict(dlr8_mcs_f1_report_full)
    dmcs9_dict_full = pd.DataFrame.from_dict(dlr9_mcs_f1_report_full)
    
    f1_matrix_train = pd.concat([lr_dict_train.transpose()['f1-score'], qda_dict_train.transpose()['f1-score'],
                   nn_dict_train.transpose()['f1-score'], rbfsvm_dict_train.transpose()['f1-score'], 
                   nb_dict_train.transpose()['f1-score'], ann_dict_train.transpose()['f1-score'], 
                   rf_dict_train.transpose()['f1-score'], ab_dict_train.transpose()['f1-score'], 
                   gb_dict_train.transpose()['f1-score'],
                   mcs3_dict_train.transpose()['f1-score'], mcs4_dict_train.transpose()['f1-score'], 
                   mcs5_dict_train.transpose()['f1-score'], mcs6_dict_train.transpose()['f1-score'],
                   mcs8_dict_train.transpose()['f1-score'], mcs9_dict_train.transpose()['f1-score'],
                   amcs3_dict_train.transpose()['f1-score'], amcs4_dict_train.transpose()['f1-score'], 
                   amcs5_dict_train.transpose()['f1-score'], amcs6_dict_train.transpose()['f1-score'],
                   amcs8_dict_train.transpose()['f1-score'], amcs9_dict_train.transpose()['f1-score'],
                   bmcs3_dict_train.transpose()['f1-score'], bmcs4_dict_train.transpose()['f1-score'], 
                   bmcs5_dict_train.transpose()['f1-score'], bmcs6_dict_train.transpose()['f1-score'],
                   bmcs8_dict_train.transpose()['f1-score'], bmcs9_dict_train.transpose()['f1-score'],
                   cmcs3_dict_train.transpose()['f1-score'], cmcs4_dict_train.transpose()['f1-score'], 
                   cmcs5_dict_train.transpose()['f1-score'], cmcs6_dict_train.transpose()['f1-score'],
                   cmcs8_dict_train.transpose()['f1-score'], cmcs9_dict_train.transpose()['f1-score'],
                   dmcs3_dict_train.transpose()['f1-score'], dmcs4_dict_train.transpose()['f1-score'], 
                   dmcs5_dict_train.transpose()['f1-score'], dmcs6_dict_train.transpose()['f1-score'],
                   dmcs8_dict_train.transpose()['f1-score'], dmcs9_dict_train.transpose()['f1-score']], axis=1)

    f1_matrix_test = pd.concat([lr_dict_test.transpose()['f1-score'], qda_dict_test.transpose()['f1-score'],
                   nn_dict_test.transpose()['f1-score'], rbfsvm_dict_test.transpose()['f1-score'], 
                   nb_dict_test.transpose()['f1-score'], ann_dict_test.transpose()['f1-score'], 
                   rf_dict_test.transpose()['f1-score'], ab_dict_test.transpose()['f1-score'], 
                   gb_dict_test.transpose()['f1-score'],
                   mcs3_dict_test.transpose()['f1-score'], mcs4_dict_test.transpose()['f1-score'], 
                   mcs5_dict_test.transpose()['f1-score'], mcs6_dict_test.transpose()['f1-score'],
                   mcs8_dict_test.transpose()['f1-score'], mcs9_dict_test.transpose()['f1-score'],
                   amcs3_dict_test.transpose()['f1-score'], amcs4_dict_test.transpose()['f1-score'], 
                   amcs5_dict_test.transpose()['f1-score'], amcs6_dict_test.transpose()['f1-score'],
                   amcs8_dict_test.transpose()['f1-score'], amcs9_dict_test.transpose()['f1-score'],
                   bmcs3_dict_test.transpose()['f1-score'], bmcs4_dict_test.transpose()['f1-score'], 
                   bmcs5_dict_test.transpose()['f1-score'], bmcs6_dict_test.transpose()['f1-score'],
                   bmcs8_dict_test.transpose()['f1-score'], bmcs9_dict_test.transpose()['f1-score'],
                   cmcs3_dict_test.transpose()['f1-score'], cmcs4_dict_test.transpose()['f1-score'], 
                   cmcs5_dict_test.transpose()['f1-score'], cmcs6_dict_test.transpose()['f1-score'],
                   cmcs8_dict_test.transpose()['f1-score'], cmcs9_dict_test.transpose()['f1-score'],
                   dmcs3_dict_test.transpose()['f1-score'], dmcs4_dict_test.transpose()['f1-score'], 
                   dmcs5_dict_test.transpose()['f1-score'], dmcs6_dict_test.transpose()['f1-score'],
                   dmcs8_dict_test.transpose()['f1-score'], dmcs9_dict_test.transpose()['f1-score']], axis=1)

    f1_matrix_full = pd.concat([lr_dict_full.transpose()['f1-score'], qda_dict_full.transpose()['f1-score'],
                   nn_dict_full.transpose()['f1-score'], rbfsvm_dict_full.transpose()['f1-score'], 
                   nb_dict_full.transpose()['f1-score'], ann_dict_full.transpose()['f1-score'], 
                   rf_dict_full.transpose()['f1-score'], ab_dict_full.transpose()['f1-score'], 
                   gb_dict_full.transpose()['f1-score'],
                   mcs3_dict_full.transpose()['f1-score'], mcs4_dict_full.transpose()['f1-score'], 
                   mcs5_dict_full.transpose()['f1-score'], mcs6_dict_full.transpose()['f1-score'],
                   mcs8_dict_full.transpose()['f1-score'], mcs9_dict_full.transpose()['f1-score'],
                   amcs3_dict_full.transpose()['f1-score'], amcs4_dict_full.transpose()['f1-score'], 
                   amcs5_dict_full.transpose()['f1-score'], amcs6_dict_full.transpose()['f1-score'],
                   amcs8_dict_full.transpose()['f1-score'], amcs9_dict_full.transpose()['f1-score'],
                   bmcs3_dict_full.transpose()['f1-score'], bmcs4_dict_full.transpose()['f1-score'], 
                   bmcs5_dict_full.transpose()['f1-score'], bmcs6_dict_full.transpose()['f1-score'],
                   bmcs8_dict_full.transpose()['f1-score'], bmcs9_dict_full.transpose()['f1-score'],
                   cmcs3_dict_full.transpose()['f1-score'], cmcs4_dict_full.transpose()['f1-score'], 
                   cmcs5_dict_full.transpose()['f1-score'], cmcs6_dict_full.transpose()['f1-score'],
                   cmcs8_dict_full.transpose()['f1-score'], cmcs9_dict_full.transpose()['f1-score'],
                   dmcs3_dict_full.transpose()['f1-score'], dmcs4_dict_full.transpose()['f1-score'], 
                   dmcs5_dict_full.transpose()['f1-score'], dmcs6_dict_full.transpose()['f1-score'],
                   dmcs8_dict_full.transpose()['f1-score'], dmcs9_dict_full.transpose()['f1-score']], axis=1)

    summary_f1_macros_train = pd.concat([summary_f1_macros_train, f1_matrix_train.iloc[[6]]], axis = 0)
    summary_f1_macros_test = pd.concat([summary_f1_macros_test, f1_matrix_test.iloc[[6]]], axis = 0)
    summary_f1_macros_full = pd.concat([summary_f1_macros_full, f1_matrix_full.iloc[[6]]], axis = 0)

    # Save Hyperparameter Data
    best_params.append(lr_best_params)
    best_params.append(qda_best_params)
    best_params.append(nn_best_params)
    best_params.append(rbfsvm_best_params)
    best_params.append(nb_best_params)
    best_params.append(ann_best_params)
    best_params.append(rf_best_params)
    best_params.append(ab_best_params)
    best_params.append(gb_best_params)
    
    dfcv = pd.DataFrame(cv_results)
    dfbp = pd.DataFrame(best_params)
    
    dfcv.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5dfcvseed'+str(seed)+'.csv', index=None, header = True)
    f1_matrix_train.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5f1trainseed'+str(seed)+'.csv', index=None, header = True)
    f1_matrix_test.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5f1testseed'+str(seed)+'.csv', index=None, header = True)
    f1_matrix_full.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5f1fullseed'+str(seed)+'.csv', index=None, header = True)
    dfbp.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5dfbpdataseed'+str(seed)+'.csv', index=None, header = True)

summary_f1_macros_train.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5summarytrainseed'+str(seed)+'.csv', index=None, header = True)
summary_f1_macros_test.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5summarytestseed'+str(seed)+'.csv', index=None, header = True)
summary_f1_macros_full.to_csv(r'C:\Users\Timothy\Dropbox\Undergrad Thesis Backups\Programming Files Dropbox\pipelineComplexMCSResults\v5summaryfullseed'+str(seed)+'.csv', index=None, header = True)
