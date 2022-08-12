# -*- coding: utf-8 -*-
"""
Created on 2022/03/15

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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline
import pyrolite.comp
import datetime

from sameClassOrderNine import sameClassOrderNine
from mcsprob import mcsprob

seeds = list(range(10, 20))

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
    
    # Ensure same k-fold splits across same seed
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=False)
    
    # Create SMOTE pipeline
    sm = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    pipeline_setup = [('sm', sm)]
        
    cv_results = []
    best_params = []
    summary_f1_macros_train = pd.DataFrame()
    summary_f1_macros_test = pd.DataFrame()
    summary_f1_macros_full = pd.DataFrame()
    summary_f1_averages_train = pd.DataFrame()
    summary_f1_averages_test = pd.DataFrame()
    summary_f1_averages_full = pd.DataFrame()
    
    # Initiate variables for models
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
    cv_results.append(gs_lr.cv_results_)        
    
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
    cv_results.append(gs_qda.cv_results_)    
    
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
    cv_results.append(gs_nn.cv_results_)    
    
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
    cv_results.append(gs_rbfsvm.cv_results_)
    
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
    cv_results.append(gs_nb.cv_results_)    
    
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
    cv_results.append(gs_ann.cv_results_)    
    
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
    cv_results.append(gs_rf.cv_results_)    
    
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
    cv_results.append(gs_ab.cv_results_)    
    
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
    cv_results.append(gs_gb.cv_results_)    
    
    print("Start MCS")
    print(datetime.datetime.now().time())
    
    # Create MCS models
    # MCS models
    # Double check to make sure all class orders are the same before creating the multi classifier system (MCS)
    matchingClassOrder = sameClassOrderNine(gs_lr, gs_qda, gs_nn, gs_rbfsvm, gs_nb, gs_ann, gs_rf, gs_ab, gs_gb)
    
    # Create the multi classifier systems (MCS)
    # Note description of MCS still says LSVM, but everytime it gets mentioned
    # it is actually not there
    
    classOrder = gs_lr.classes_
    if matchingClassOrder:
        # MCS2a with top 2 [ab, gb]
        mcs2apredtrain = mcsprob(classOrder, ab_probs_train, gb_probs_train)
        mcs2atrain_report = classification_report(y_train_final, mcs2apredtrain, output_dict=True)
        # MCS2b with top 2, no forest repetition [rbfsvm, gb]
        mcs2bpredtrain = mcsprob(classOrder, rbfsvm_probs_train, gb_probs_train)
        mcs2btrain_report = classification_report(y_train_final, mcs2bpredtrain, output_dict=True)
        # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
        mcs3apredtrain = mcsprob(classOrder, rbfsvm_probs_train, ann_probs_train, gb_probs_train)
        mcs3atrain_report = classification_report(y_train_final, mcs3apredtrain, output_dict=True)
        # MCS3b forest repetition [rf, ab, gb]
        mcs3bpredtrain = mcsprob(classOrder, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs3btrain_report = classification_report(y_train_final, mcs3bpredtrain, output_dict=True)
        # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
        mcs5predtrain = mcsprob(classOrder, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs5train_report = classification_report(y_train_final, mcs5predtrain, output_dict=True)
        # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
        mcs6predtrain = mcsprob(classOrder, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs6train_report = classification_report(y_train_final, mcs6predtrain, output_dict=True)
        # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
        mcs7predtrain = mcsprob(classOrder, qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs7train_report = classification_report(y_train_final, mcs7predtrain, output_dict=True)
        # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
        mcs8predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs8train_report = classification_report(y_train_final, mcs8predtrain, output_dict=True)
        # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
        mcs9predtrain = mcsprob(classOrder, lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, nb_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train)
        mcs9train_report = classification_report(y_train_final, mcs9predtrain, output_dict=True)
        
        # MCS2a with top 2 [ab, gb]
        mcs2apredtest = mcsprob(classOrder, ab_probs_test, gb_probs_test)
        mcs2atest_report = classification_report(y_test_final, mcs2apredtest, output_dict=True)
        # MCS2b with top 2, no forest repetition [rbfsvm, gb]
        mcs2bpredtest = mcsprob(classOrder, rbfsvm_probs_test, gb_probs_test)
        mcs2btest_report = classification_report(y_test_final, mcs2bpredtest, output_dict=True)
        # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
        mcs3apredtest = mcsprob(classOrder, rbfsvm_probs_test, ann_probs_test, gb_probs_test)
        mcs3atest_report = classification_report(y_test_final, mcs3apredtest, output_dict=True)
        # MCS3b forest repetition [rf, ab, gb]
        mcs3bpredtest = mcsprob(classOrder, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs3btest_report = classification_report(y_test_final, mcs3bpredtest, output_dict=True)
        # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
        mcs5predtest = mcsprob(classOrder, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs5test_report = classification_report(y_test_final, mcs5predtest, output_dict=True)
        # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
        mcs6predtest = mcsprob(classOrder, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs6test_report = classification_report(y_test_final, mcs6predtest, output_dict=True)
        # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
        mcs7predtest = mcsprob(classOrder, qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs7test_report = classification_report(y_test_final, mcs7predtest, output_dict=True)
        # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
        mcs8predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs8test_report = classification_report(y_test_final, mcs8predtest, output_dict=True)
        # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
        mcs9predtest = mcsprob(classOrder, lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, nb_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test)
        mcs9test_report = classification_report(y_test_final, mcs9predtest, output_dict=True)
        
        # MCS2a with top 2 [ab, gb]
        mcs2apredfull = mcsprob(classOrder, ab_probs_full, gb_probs_full)
        mcs2afull_report = classification_report(y_full_final, mcs2apredfull, output_dict=True)
        # MCS2b with top 2, no forest repetition [rbfsvm, gb]
        mcs2bpredfull = mcsprob(classOrder, rbfsvm_probs_full, gb_probs_full)
        mcs2bfull_report = classification_report(y_full_final, mcs2bpredfull, output_dict=True)
        # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
        mcs3apredfull = mcsprob(classOrder, rbfsvm_probs_full, ann_probs_full, gb_probs_full)
        mcs3afull_report = classification_report(y_full_final, mcs3apredfull, output_dict=True)
        # MCS3b forest repetition [rf, ab, gb]
        mcs3bpredfull = mcsprob(classOrder, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs3bfull_report = classification_report(y_full_final, mcs3bpredfull, output_dict=True)
        # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
        mcs5predfull = mcsprob(classOrder, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs5full_report = classification_report(y_full_final, mcs5predfull, output_dict=True)
        # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
        mcs6predfull = mcsprob(classOrder, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs6full_report = classification_report(y_full_final, mcs6predfull, output_dict=True)
        # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
        mcs7predfull = mcsprob(classOrder, qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs7full_report = classification_report(y_full_final, mcs7predfull, output_dict=True)
        # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
        mcs8predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs8full_report = classification_report(y_full_final, mcs8predfull, output_dict=True)
        # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
        mcs9predfull = mcsprob(classOrder, lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, nb_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full)
        mcs9full_report = classification_report(y_full_final, mcs9predfull, output_dict=True)
        
    else:
        print('Failed matching class order')
    
    # AMCS: Applies LR to the imbalanced train set
    
    # MCS2a with top 2 [ab, gb]
    lr2a_train = np.concatenate((ab_probs_train, gb_probs_train), axis=1)
    lr2a_test = np.concatenate((ab_probs_test, gb_probs_test), axis=1)
    lr2a_full = np.concatenate((ab_probs_full, gb_probs_full), axis=1)

    alr2a_mcs = LogisticRegression(random_state = seed)
    alr2a_mcs.fit(lr2a_train, y_train_final)
    alr2a_mcs_pred_train = alr2a_mcs.predict(lr2a_train)
    alrmcs2atrain_report = classification_report(y_train_final, alr2a_mcs_pred_train, output_dict=True)
    alr2a_mcs_pred_test = alr2a_mcs.predict(lr2a_test)
    alrmcs2atest_report = classification_report(y_test_final, alr2a_mcs_pred_test, output_dict=True)
    alr2a_mcs_pred_full = alr2a_mcs.predict(lr2a_full)
    alrmcs2afull_report = classification_report(y_full_final, alr2a_mcs_pred_full, output_dict=True)
    
    # MCS2b with top 2, no forest repetition [rbfsvm, gb]
    lr2b_train = np.concatenate((rbfsvm_probs_train, gb_probs_train), axis=1)
    lr2b_test = np.concatenate((rbfsvm_probs_test, gb_probs_test), axis=1)
    lr2b_full = np.concatenate((rbfsvm_probs_full, gb_probs_full), axis=1)

    alr2b_mcs = LogisticRegression(random_state = seed)
    alr2b_mcs.fit(lr2b_train, y_train_final)
    alr2b_mcs_pred_train = alr2b_mcs.predict(lr2b_train)
    alrmcs2btrain_report = classification_report(y_train_final, alr2b_mcs_pred_train, output_dict=True)
    alr2b_mcs_pred_test = alr2b_mcs.predict(lr2b_test)
    alrmcs2btest_report = classification_report(y_test_final, alr2b_mcs_pred_test, output_dict=True)
    alr2b_mcs_pred_full = alr2b_mcs.predict(lr2b_full)
    alrmcs2bfull_report = classification_report(y_full_final, alr2b_mcs_pred_full, output_dict=True)
    
    # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
    lr3a_train = np.concatenate((rbfsvm_probs_train, ann_probs_train, gb_probs_train), axis=1)
    lr3a_test = np.concatenate((rbfsvm_probs_test, ann_probs_test, gb_probs_test), axis=1)
    lr3a_full = np.concatenate((rbfsvm_probs_full, ann_probs_full, gb_probs_full), axis=1)

    alr3a_mcs = LogisticRegression(random_state = seed)
    alr3a_mcs.fit(lr3a_train, y_train_final)
    alr3a_mcs_pred_train = alr3a_mcs.predict(lr3a_train)
    alrmcs3atrain_report = classification_report(y_train_final, alr3a_mcs_pred_train, output_dict=True)
    alr3a_mcs_pred_test = alr3a_mcs.predict(lr3a_test)
    alrmcs3atest_report = classification_report(y_test_final, alr3a_mcs_pred_test, output_dict=True)
    alr3a_mcs_pred_full = alr3a_mcs.predict(lr3a_full)
    alrmcs3afull_report = classification_report(y_full_final, alr3a_mcs_pred_full, output_dict=True)
    
    # MCS3b forest repetition [rf, ab, gb]
    lr3b_train = np.concatenate((rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr3b_test = np.concatenate((rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr3b_full = np.concatenate((rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr3b_mcs = LogisticRegression(random_state = seed)
    alr3b_mcs.fit(lr3b_train, y_train_final)
    alr3b_mcs_pred_train = alr3b_mcs.predict(lr3b_train)
    alrmcs3btrain_report = classification_report(y_train_final, alr3b_mcs_pred_train, output_dict=True)
    alr3b_mcs_pred_test = alr3b_mcs.predict(lr3b_test)
    alrmcs3btest_report = classification_report(y_test_final, alr3b_mcs_pred_test, output_dict=True)
    alr3b_mcs_pred_full = alr3b_mcs.predict(lr3b_full)
    alrmcs3bfull_report = classification_report(y_full_final, alr3b_mcs_pred_full, output_dict=True)
    
    # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
    lr5_train = np.concatenate((rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr5_test = np.concatenate((rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr5_full = np.concatenate((rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr5_mcs = LogisticRegression(random_state = seed)
    alr5_mcs.fit(lr5_train, y_train_final)
    alr5_mcs_pred_train = alr5_mcs.predict(lr5_train)
    alrmcs5train_report = classification_report(y_train_final, alr5_mcs_pred_train, output_dict=True)
    alr5_mcs_pred_test = alr5_mcs.predict(lr5_test)
    alrmcs5test_report = classification_report(y_test_final, alr5_mcs_pred_test, output_dict=True)
    alr5_mcs_pred_full = alr5_mcs.predict(lr5_full)
    alrmcs5full_report = classification_report(y_full_final, alr5_mcs_pred_full, output_dict=True)
    
    # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
    lr6_train = np.concatenate((nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr6_test = np.concatenate((nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr6_full = np.concatenate((nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr6_mcs = LogisticRegression(random_state = seed)
    alr6_mcs.fit(lr6_train, y_train_final)
    alr6_mcs_pred_train = alr6_mcs.predict(lr6_train)
    alrmcs6train_report = classification_report(y_train_final, alr6_mcs_pred_train, output_dict=True)
    alr6_mcs_pred_test = alr6_mcs.predict(lr6_test)
    alrmcs6test_report = classification_report(y_test_final, alr6_mcs_pred_test, output_dict=True)
    alr6_mcs_pred_full = alr6_mcs.predict(lr6_full)
    alrmcs6full_report = classification_report(y_full_final, alr6_mcs_pred_full, output_dict=True)
    
    # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
    lr7_train = np.concatenate((qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr7_test = np.concatenate((qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr7_full = np.concatenate((qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr7_mcs = LogisticRegression(random_state = seed)
    alr7_mcs.fit(lr7_train, y_train_final)
    alr7_mcs_pred_train = alr7_mcs.predict(lr7_train)
    alrmcs7train_report = classification_report(y_train_final, alr7_mcs_pred_train, output_dict=True)
    alr7_mcs_pred_test = alr7_mcs.predict(lr7_test)
    alrmcs7test_report = classification_report(y_test_final, alr7_mcs_pred_test, output_dict=True)
    alr7_mcs_pred_full = alr7_mcs.predict(lr7_full)
    alrmcs7full_report = classification_report(y_full_final, alr7_mcs_pred_full, output_dict=True)
    
    # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
    lr8_train = np.concatenate((lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr8_test = np.concatenate((lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr8_full = np.concatenate((lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr8_mcs = LogisticRegression(random_state = seed)
    alr8_mcs.fit(lr8_train, y_train_final)
    alr8_mcs_pred_train = alr8_mcs.predict(lr8_train)
    alrmcs8train_report = classification_report(y_train_final, alr8_mcs_pred_train, output_dict=True)
    alr8_mcs_pred_test = alr8_mcs.predict(lr8_test)
    alrmcs8test_report = classification_report(y_test_final, alr8_mcs_pred_test, output_dict=True)
    alr8_mcs_pred_full = alr8_mcs.predict(lr8_full)
    alrmcs8full_report = classification_report(y_full_final, alr8_mcs_pred_full, output_dict=True)
    
    # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
    lr9_train = np.concatenate((lr_probs_train, qda_probs_train, nn_probs_train, rbfsvm_probs_train, nb_probs_train, ann_probs_train, rf_probs_train, ab_probs_train, gb_probs_train), axis=1)
    lr9_test = np.concatenate((lr_probs_test, qda_probs_test, nn_probs_test, rbfsvm_probs_test, nb_probs_test, ann_probs_test, rf_probs_test, ab_probs_test, gb_probs_test), axis=1)
    lr9_full = np.concatenate((lr_probs_full, qda_probs_full, nn_probs_full, rbfsvm_probs_full, nb_probs_full, ann_probs_full, rf_probs_full, ab_probs_full, gb_probs_full), axis=1)

    alr9_mcs = LogisticRegression(random_state = seed)
    alr9_mcs.fit(lr9_train, y_train_final)
    alr9_mcs_pred_train = alr9_mcs.predict(lr9_train)
    alrmcs9train_report = classification_report(y_train_final, alr9_mcs_pred_train, output_dict=True)
    alr9_mcs_pred_test = alr9_mcs.predict(lr9_test)
    alrmcs9test_report = classification_report(y_test_final, alr9_mcs_pred_test, output_dict=True)
    alr9_mcs_pred_full = alr9_mcs.predict(lr9_full)
    alrmcs9full_report = classification_report(y_full_final, alr9_mcs_pred_full, output_dict=True)


    # BMCS: Applies LR to a SMOTE-ed data set of train set
    
    # MCS2a with top 2 [ab, gb]
    sm2a = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b2a, y_train_b2a = sm2a.fit_resample(lr2a_train, y_train_final)
    
    blr2a_mcs = LogisticRegression(random_state = seed)
    blr2a_mcs.fit(X_train_b2a, y_train_b2a)
    blr2a_mcs_pred_train = blr2a_mcs.predict(lr2a_train)
    blrmcs2atrain_report = classification_report(y_train_final, blr2a_mcs_pred_train, output_dict=True)
    blr2a_mcs_pred_test = blr2a_mcs.predict(lr2a_test)
    blrmcs2atest_report = classification_report(y_test_final, blr2a_mcs_pred_test, output_dict=True)
    blr2a_mcs_pred_full = blr2a_mcs.predict(lr2a_full)
    blrmcs2afull_report = classification_report(y_full_final, blr2a_mcs_pred_full, output_dict=True)
    
    # MCS2b with top 2, no forest repetition [rbfsvm, gb]
    sm2b = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b2b, y_train_b2b = sm2b.fit_resample(lr2b_train, y_train_final)
    
    blr2b_mcs = LogisticRegression(random_state = seed)
    blr2b_mcs.fit(X_train_b2b, y_train_b2b)
    blr2b_mcs_pred_train = blr2b_mcs.predict(lr2b_train)
    blrmcs2btrain_report = classification_report(y_train_final, blr2b_mcs_pred_train, output_dict=True)
    blr2b_mcs_pred_test = blr2b_mcs.predict(lr2b_test)
    blrmcs2btest_report = classification_report(y_test_final, blr2b_mcs_pred_test, output_dict=True)
    blr2b_mcs_pred_full = blr2b_mcs.predict(lr2b_full)
    blrmcs2bfull_report = classification_report(y_full_final, blr2b_mcs_pred_full, output_dict=True)
    
    # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
    sm3a = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b3a, y_train_b3a = sm3a.fit_resample(lr3a_train, y_train_final)
    
    blr3a_mcs = LogisticRegression(random_state = seed)
    blr3a_mcs.fit(X_train_b3a, y_train_b3a)
    blr3a_mcs_pred_train = blr3a_mcs.predict(lr3a_train)
    blrmcs3atrain_report = classification_report(y_train_final, blr3a_mcs_pred_train, output_dict=True)
    blr3a_mcs_pred_test = blr3a_mcs.predict(lr3a_test)
    blrmcs3atest_report = classification_report(y_test_final, blr3a_mcs_pred_test, output_dict=True)
    blr3a_mcs_pred_full = blr3a_mcs.predict(lr3a_full)
    blrmcs3afull_report = classification_report(y_full_final, blr3a_mcs_pred_full, output_dict=True)
    
    # MCS3b forest repetition [rf, ab, gb]
    sm3b = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b3b, y_train_b3b = sm3b.fit_resample(lr3b_train, y_train_final)
    
    blr3b_mcs = LogisticRegression(random_state = seed)
    blr3b_mcs.fit(X_train_b3b, y_train_b3b)
    blr3b_mcs_pred_train = blr3b_mcs.predict(lr3b_train)
    blrmcs3btrain_report = classification_report(y_train_final, blr3b_mcs_pred_train, output_dict=True)
    blr3b_mcs_pred_test = blr3b_mcs.predict(lr3b_test)
    blrmcs3btest_report = classification_report(y_test_final, blr3b_mcs_pred_test, output_dict=True)
    blr3b_mcs_pred_full = blr3b_mcs.predict(lr3b_full)
    blrmcs3bfull_report = classification_report(y_full_final, blr3b_mcs_pred_full, output_dict=True)
    
    # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
    sm5 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b5, y_train_b5 = sm5.fit_resample(lr5_train, y_train_final)
    
    blr5_mcs = LogisticRegression(random_state = seed)
    blr5_mcs.fit(X_train_b5, y_train_b5)
    blr5_mcs_pred_train = blr5_mcs.predict(lr5_train)
    blrmcs5train_report = classification_report(y_train_final, blr5_mcs_pred_train, output_dict=True)
    blr5_mcs_pred_test = blr5_mcs.predict(lr5_test)
    blrmcs5test_report = classification_report(y_test_final, blr5_mcs_pred_test, output_dict=True)
    blr5_mcs_pred_full = blr5_mcs.predict(lr5_full)
    blrmcs5full_report = classification_report(y_full_final, blr5_mcs_pred_full, output_dict=True)
    
    # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
    sm6 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b6, y_train_b6 = sm6.fit_resample(lr6_train, y_train_final)
    
    blr6_mcs = LogisticRegression(random_state = seed)
    blr6_mcs.fit(X_train_b6, y_train_b6)
    blr6_mcs_pred_train = blr6_mcs.predict(lr6_train)
    blrmcs6train_report = classification_report(y_train_final, blr6_mcs_pred_train, output_dict=True)
    blr6_mcs_pred_test = blr6_mcs.predict(lr6_test)
    blrmcs6test_report = classification_report(y_test_final, blr6_mcs_pred_test, output_dict=True)
    blr6_mcs_pred_full = blr6_mcs.predict(lr6_full)
    blrmcs6full_report = classification_report(y_full_final, blr6_mcs_pred_full, output_dict=True)
    
    # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
    sm7 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b7, y_train_b7 = sm7.fit_resample(lr7_train, y_train_final)
    
    blr7_mcs = LogisticRegression(random_state = seed)
    blr7_mcs.fit(X_train_b7, y_train_b7)
    blr7_mcs_pred_train = blr7_mcs.predict(lr7_train)
    blrmcs7train_report = classification_report(y_train_final, blr7_mcs_pred_train, output_dict=True)
    blr7_mcs_pred_test = blr7_mcs.predict(lr7_test)
    blrmcs7test_report = classification_report(y_test_final, blr7_mcs_pred_test, output_dict=True)
    blr7_mcs_pred_full = blr7_mcs.predict(lr7_full)
    blrmcs7full_report = classification_report(y_full_final, blr7_mcs_pred_full, output_dict=True)
    
    # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
    sm8 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b8, y_train_b8 = sm8.fit_resample(lr8_train, y_train_final)
    
    blr8_mcs = LogisticRegression(random_state = seed)
    blr8_mcs.fit(X_train_b8, y_train_b8)
    blr8_mcs_pred_train = blr8_mcs.predict(lr8_train)
    blrmcs8train_report = classification_report(y_train_final, blr8_mcs_pred_train, output_dict=True)
    blr8_mcs_pred_test = blr8_mcs.predict(lr8_test)
    blrmcs8test_report = classification_report(y_test_final, blr8_mcs_pred_test, output_dict=True)
    blr8_mcs_pred_full = blr8_mcs.predict(lr8_full)
    blrmcs8full_report = classification_report(y_full_final, blr8_mcs_pred_full, output_dict=True)
    
    # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
    sm9 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_b9, y_train_b9 = sm9.fit_resample(lr9_train, y_train_final)
    
    blr9_mcs = LogisticRegression(random_state = seed)
    blr9_mcs.fit(X_train_b9, y_train_b9)
    blr9_mcs_pred_train = blr9_mcs.predict(lr9_train)
    blrmcs9train_report = classification_report(y_train_final, blr9_mcs_pred_train, output_dict=True)
    blr9_mcs_pred_test = blr9_mcs.predict(lr9_test)
    blrmcs9test_report = classification_report(y_test_final, blr9_mcs_pred_test, output_dict=True)
    blr9_mcs_pred_full = blr9_mcs.predict(lr9_full)
    blrmcs9full_report = classification_report(y_full_final, blr9_mcs_pred_full, output_dict=True)


    # CMCS: Applies LR to a scaled data set of train set

    # MCS2a with top 2 [ab, gb]
    scal2a = StandardScaler()
    X_train_c2a = scaler.fit_transform(lr2a_train)
    X_test_c2a = scaler.transform(lr2a_test)
    X_full_c2a = scaler.transform(lr2a_full)
    
    clr2a_mcs = LogisticRegression(random_state = seed)
    clr2a_mcs.fit(X_train_c2a, y_train_final)
    clr2a_mcs_pred_train = clr2a_mcs.predict(X_train_c2a)
    clrmcs2atrain_report = classification_report(y_train_final, clr2a_mcs_pred_train, output_dict=True)
    clr2a_mcs_pred_test = clr2a_mcs.predict(X_test_c2a)
    clrmcs2atest_report = classification_report(y_test_final, clr2a_mcs_pred_test, output_dict=True)
    clr2a_mcs_pred_full = clr2a_mcs.predict(X_full_c2a)
    clrmcs2afull_report = classification_report(y_full_final, clr2a_mcs_pred_full, output_dict=True)
    
    # MCS2b with top 2, no forest repetition [rbfsvm, gb]
    scal2b = StandardScaler()
    X_train_c2b = scaler.fit_transform(lr2b_train)
    X_test_c2b = scaler.transform(lr2b_test)
    X_full_c2b = scaler.transform(lr2b_full)
    
    clr2b_mcs = LogisticRegression(random_state = seed)
    clr2b_mcs.fit(X_train_c2b, y_train_final)
    clr2b_mcs_pred_train = clr2b_mcs.predict(X_train_c2b)
    clrmcs2btrain_report = classification_report(y_train_final, clr2b_mcs_pred_train, output_dict=True)
    clr2b_mcs_pred_test = clr2b_mcs.predict(X_test_c2b)
    clrmcs2btest_report = classification_report(y_test_final, clr2b_mcs_pred_test, output_dict=True)
    clr2b_mcs_pred_full = clr2b_mcs.predict(X_full_c2b)
    clrmcs2bfull_report = classification_report(y_full_final, clr2b_mcs_pred_full, output_dict=True)
    
    # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
    scal3a = StandardScaler()
    X_train_c3a = scaler.fit_transform(lr3a_train)
    X_test_c3a = scaler.transform(lr3a_test)
    X_full_c3a = scaler.transform(lr3a_full)
    
    clr3a_mcs = LogisticRegression(random_state = seed)
    clr3a_mcs.fit(X_train_c3a, y_train_final)
    clr3a_mcs_pred_train = clr3a_mcs.predict(X_train_c3a)
    clrmcs3atrain_report = classification_report(y_train_final, clr3a_mcs_pred_train, output_dict=True)
    clr3a_mcs_pred_test = clr3a_mcs.predict(X_test_c3a)
    clrmcs3atest_report = classification_report(y_test_final, clr3a_mcs_pred_test, output_dict=True)
    clr3a_mcs_pred_full = clr3a_mcs.predict(X_full_c3a)
    clrmcs3afull_report = classification_report(y_full_final, clr3a_mcs_pred_full, output_dict=True)
    
    # MCS3b forest repetition [rf, ab, gb]
    scal3b = StandardScaler()
    X_train_c3b = scaler.fit_transform(lr3b_train)
    X_test_c3b = scaler.transform(lr3b_test)
    X_full_c3b = scaler.transform(lr3b_full)
    
    clr3b_mcs = LogisticRegression(random_state = seed)
    clr3b_mcs.fit(X_train_c3b, y_train_final)
    clr3b_mcs_pred_train = clr3b_mcs.predict(X_train_c3b)
    clrmcs3btrain_report = classification_report(y_train_final, clr3b_mcs_pred_train, output_dict=True)
    clr3b_mcs_pred_test = clr3b_mcs.predict(X_test_c3b)
    clrmcs3btest_report = classification_report(y_test_final, clr3b_mcs_pred_test, output_dict=True)
    clr3b_mcs_pred_full = clr3b_mcs.predict(X_full_c3b)
    clrmcs3bfull_report = classification_report(y_full_final, clr3b_mcs_pred_full, output_dict=True)
    
    # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
    scal5 = StandardScaler()
    X_train_c5 = scaler.fit_transform(lr5_train)
    X_test_c5 = scaler.transform(lr5_test)
    X_full_c5 = scaler.transform(lr5_full)
    
    clr5_mcs = LogisticRegression(random_state = seed)
    clr5_mcs.fit(X_train_c5, y_train_final)
    clr5_mcs_pred_train = clr5_mcs.predict(X_train_c5)
    clrmcs5train_report = classification_report(y_train_final, clr5_mcs_pred_train, output_dict=True)
    clr5_mcs_pred_test = clr5_mcs.predict(X_test_c5)
    clrmcs5test_report = classification_report(y_test_final, clr5_mcs_pred_test, output_dict=True)
    clr5_mcs_pred_full = clr5_mcs.predict(X_full_c5)
    clrmcs5full_report = classification_report(y_full_final, clr5_mcs_pred_full, output_dict=True)
    
    # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
    scal6 = StandardScaler()
    X_train_c6 = scaler.fit_transform(lr6_train)
    X_test_c6 = scaler.transform(lr6_test)
    X_full_c6 = scaler.transform(lr6_full)
    
    clr6_mcs = LogisticRegression(random_state = seed)
    clr6_mcs.fit(X_train_c6, y_train_final)
    clr6_mcs_pred_train = clr6_mcs.predict(X_train_c6)
    clrmcs6train_report = classification_report(y_train_final, clr6_mcs_pred_train, output_dict=True)
    clr6_mcs_pred_test = clr6_mcs.predict(X_test_c6)
    clrmcs6test_report = classification_report(y_test_final, clr6_mcs_pred_test, output_dict=True)
    clr6_mcs_pred_full = clr6_mcs.predict(X_full_c6)
    clrmcs6full_report = classification_report(y_full_final, clr6_mcs_pred_full, output_dict=True)
    
    # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
    scal7 = StandardScaler()
    X_train_c7 = scaler.fit_transform(lr7_train)
    X_test_c7 = scaler.transform(lr7_test)
    X_full_c7 = scaler.transform(lr7_full)
    
    clr7_mcs = LogisticRegression(random_state = seed)
    clr7_mcs.fit(X_train_c7, y_train_final)
    clr7_mcs_pred_train = clr7_mcs.predict(X_train_c7)
    clrmcs7train_report = classification_report(y_train_final, clr7_mcs_pred_train, output_dict=True)
    clr7_mcs_pred_test = clr7_mcs.predict(X_test_c7)
    clrmcs7test_report = classification_report(y_test_final, clr7_mcs_pred_test, output_dict=True)
    clr7_mcs_pred_full = clr7_mcs.predict(X_full_c7)
    clrmcs7full_report = classification_report(y_full_final, clr7_mcs_pred_full, output_dict=True)
    
    # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
    scal8 = StandardScaler()
    X_train_c8 = scaler.fit_transform(lr8_train)
    X_test_c8 = scaler.transform(lr8_test)
    X_full_c8 = scaler.transform(lr8_full)
    
    clr8_mcs = LogisticRegression(random_state = seed)
    clr8_mcs.fit(X_train_c8, y_train_final)
    clr8_mcs_pred_train = clr8_mcs.predict(X_train_c8)
    clrmcs8train_report = classification_report(y_train_final, clr8_mcs_pred_train, output_dict=True)
    clr8_mcs_pred_test = clr8_mcs.predict(X_test_c8)
    clrmcs8test_report = classification_report(y_test_final, clr8_mcs_pred_test, output_dict=True)
    clr8_mcs_pred_full = clr8_mcs.predict(X_full_c8)
    clrmcs8full_report = classification_report(y_full_final, clr8_mcs_pred_full, output_dict=True)
    
    # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
    scal9 = StandardScaler()
    X_train_c9 = scaler.fit_transform(lr9_train)
    X_test_c9 = scaler.transform(lr9_test)
    X_full_c9 = scaler.transform(lr9_full)
    
    clr9_mcs = LogisticRegression(random_state = seed)
    clr9_mcs.fit(X_train_c9, y_train_final)
    clr9_mcs_pred_train = clr9_mcs.predict(X_train_c9)
    clrmcs9train_report = classification_report(y_train_final, clr9_mcs_pred_train, output_dict=True)
    clr9_mcs_pred_test = clr9_mcs.predict(X_test_c9)
    clrmcs9test_report = classification_report(y_test_final, clr9_mcs_pred_test, output_dict=True)
    clr9_mcs_pred_full = clr9_mcs.predict(X_full_c9)
    clrmcs9full_report = classification_report(y_full_final, clr9_mcs_pred_full, output_dict=True)

    
    # DMCS: Applies LR to a SMOTE-ed Scaled data set of train set
    
    # MCS2a with top 2 [ab, gb]
    smd2a = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d2a, y_train_d2a = smd2a.fit_resample(X_train_c2a, y_train_og)
    
    dlr2a_mcs = LogisticRegression(random_state = seed)
    dlr2a_mcs.fit(X_train_d2a, y_train_d2a)
    dlr2a_mcs_pred_train = dlr2a_mcs.predict(X_train_c2a)
    dlrmcs2atrain_report = classification_report(y_train_final, dlr2a_mcs_pred_train, output_dict=True)
    dlr2a_mcs_pred_test = dlr2a_mcs.predict(X_test_c2a)
    dlrmcs2atest_report = classification_report(y_test_final, dlr2a_mcs_pred_test, output_dict=True)
    dlr2a_mcs_pred_full = dlr2a_mcs.predict(X_full_c2a)
    dlrmcs2afull_report = classification_report(y_full_final, dlr2a_mcs_pred_full, output_dict=True)
    
    # MCS2b with top 2, no forest repetition [rbfsvm, gb]
    smd2b = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d2b, y_train_d2b = smd2b.fit_resample(X_train_c2b, y_train_og)
    
    dlr2b_mcs = LogisticRegression(random_state = seed)
    dlr2b_mcs.fit(X_train_d2b, y_train_d2b)
    dlr2b_mcs_pred_train = dlr2b_mcs.predict(X_train_c2b)
    dlrmcs2btrain_report = classification_report(y_train_final, dlr2b_mcs_pred_train, output_dict=True)
    dlr2b_mcs_pred_test = dlr2b_mcs.predict(X_test_c2b)
    dlrmcs2btest_report = classification_report(y_test_final, dlr2b_mcs_pred_test, output_dict=True)
    dlr2b_mcs_pred_full = dlr2b_mcs.predict(X_full_c2b)
    dlrmcs2bfull_report = classification_report(y_full_final, dlr2b_mcs_pred_full, output_dict=True)
    
    # MCS3a with top 3, no forest repetition [rbfsvm, ann, gb]
    smd3a = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d3a, y_train_d3a = smd3a.fit_resample(X_train_c3a, y_train_og)
    
    dlr3a_mcs = LogisticRegression(random_state = seed)
    dlr3a_mcs.fit(X_train_d3a, y_train_d3a)
    dlr3a_mcs_pred_train = dlr3a_mcs.predict(X_train_c3a)
    dlrmcs3atrain_report = classification_report(y_train_final, dlr3a_mcs_pred_train, output_dict=True)
    dlr3a_mcs_pred_test = dlr3a_mcs.predict(X_test_c3a)
    dlrmcs3atest_report = classification_report(y_test_final, dlr3a_mcs_pred_test, output_dict=True)
    dlr3a_mcs_pred_full = dlr3a_mcs.predict(X_full_c3a)
    dlrmcs3afull_report = classification_report(y_full_final, dlr3a_mcs_pred_full, output_dict=True)
    
    # MCS3b forest repetition [rf, ab, gb]
    smd3b = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d3b, y_train_d3b = smd3b.fit_resample(X_train_c3b, y_train_og)
    
    dlr3b_mcs = LogisticRegression(random_state = seed)
    dlr3b_mcs.fit(X_train_d3b, y_train_d3b)
    dlr3b_mcs_pred_train = dlr3b_mcs.predict(X_train_c3b)
    dlrmcs3btrain_report = classification_report(y_train_final, dlr3b_mcs_pred_train, output_dict=True)
    dlr3b_mcs_pred_test = dlr3b_mcs.predict(X_test_c3b)
    dlrmcs3btest_report = classification_report(y_test_final, dlr3b_mcs_pred_test, output_dict=True)
    dlr3b_mcs_pred_full = dlr3b_mcs.predict(X_full_c3b)
    dlrmcs3bfull_report = classification_report(y_full_final, dlr3b_mcs_pred_full, output_dict=True)
    
    # MCS5 with top 5 [rbfsvm, ann, rf, ab, gb]
    smd5 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d5, y_train_d5 = smd5.fit_resample(X_train_c5, y_train_og)
    
    dlr5_mcs = LogisticRegression(random_state = seed)
    dlr5_mcs.fit(X_train_d5, y_train_d5)
    dlr5_mcs_pred_train = dlr5_mcs.predict(X_train_c5)
    dlrmcs5train_report = classification_report(y_train_final, dlr5_mcs_pred_train, output_dict=True)
    dlr5_mcs_pred_test = dlr5_mcs.predict(X_test_c5)
    dlrmcs5test_report = classification_report(y_test_final, dlr5_mcs_pred_test, output_dict=True)
    dlr5_mcs_pred_full = dlr5_mcs.predict(X_full_c5)
    dlrmcs5full_report = classification_report(y_full_final, dlr5_mcs_pred_full, output_dict=True)
    
    # MCS6 with top 6, [nn, rbfsvm, ann, rf, ab, gb]
    smd6 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d6, y_train_d6 = smd6.fit_resample(X_train_c6, y_train_og)
    
    dlr6_mcs = LogisticRegression(random_state = seed)
    dlr6_mcs.fit(X_train_d6, y_train_d6)
    dlr6_mcs_pred_train = dlr6_mcs.predict(X_train_c6)
    dlrmcs6train_report = classification_report(y_train_final, dlr6_mcs_pred_train, output_dict=True)
    dlr6_mcs_pred_test = dlr6_mcs.predict(X_test_c6)
    dlrmcs6test_report = classification_report(y_test_final, dlr6_mcs_pred_test, output_dict=True)
    dlr6_mcs_pred_full = dlr6_mcs.predict(X_full_c6)
    dlrmcs6full_report = classification_report(y_full_final, dlr6_mcs_pred_full, output_dict=True)
    
    # MCS7 with top 7, [qda, nn, rbfsvm, ann, rf, ab, gb]
    smd7 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d7, y_train_d7 = smd7.fit_resample(X_train_c7, y_train_og)
    
    dlr7_mcs = LogisticRegression(random_state = seed)
    dlr7_mcs.fit(X_train_d7, y_train_d7)
    dlr7_mcs_pred_train = dlr7_mcs.predict(X_train_c7)
    dlrmcs7train_report = classification_report(y_train_final, dlr7_mcs_pred_train, output_dict=True)
    dlr7_mcs_pred_test = dlr7_mcs.predict(X_test_c7)
    dlrmcs7test_report = classification_report(y_test_final, dlr7_mcs_pred_test, output_dict=True)
    dlr7_mcs_pred_full = dlr7_mcs.predict(X_full_c7)
    dlrmcs7full_report = classification_report(y_full_final, dlr7_mcs_pred_full, output_dict=True)
    
    # MCS8 with top 8, [lr, qda, nn, rbfsvm, ann, rf, ab, gb]
    smd8 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d8, y_train_d8 = smd8.fit_resample(X_train_c8, y_train_og)
    
    dlr8_mcs = LogisticRegression(random_state = seed)
    dlr8_mcs.fit(X_train_d8, y_train_d8)
    dlr8_mcs_pred_train = dlr8_mcs.predict(X_train_c8)
    dlrmcs8train_report = classification_report(y_train_final, dlr8_mcs_pred_train, output_dict=True)
    dlr8_mcs_pred_test = dlr8_mcs.predict(X_test_c8)
    dlrmcs8test_report = classification_report(y_test_final, dlr8_mcs_pred_test, output_dict=True)
    dlr8_mcs_pred_full = dlr8_mcs.predict(X_full_c8)
    dlrmcs8full_report = classification_report(y_full_final, dlr8_mcs_pred_full, output_dict=True)
    
    # MCS9 with 9, all models [lr, qda, nn, rbfsvm, nb, ann, rf, ab, gb]
    smd9 = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed)
    X_train_d9, y_train_d9 = smd9.fit_resample(X_train_c9, y_train_og)
    
    dlr9_mcs = LogisticRegression(random_state = seed)
    dlr9_mcs.fit(X_train_d9, y_train_d9)
    dlr9_mcs_pred_train = dlr9_mcs.predict(X_train_c9)
    dlrmcs9train_report = classification_report(y_train_final, dlr9_mcs_pred_train, output_dict=True)
    dlr9_mcs_pred_test = dlr9_mcs.predict(X_test_c9)
    dlrmcs9test_report = classification_report(y_test_final, dlr9_mcs_pred_test, output_dict=True)
    dlr9_mcs_pred_full = dlr9_mcs.predict(X_full_c9)
    dlrmcs9full_report = classification_report(y_full_final, dlr9_mcs_pred_full, output_dict=True)
    
    
    print("Stop MCS")
    print(datetime.datetime.now().time())
    
    
    
    # Create the trivial classifier
    # The most common unit is Whitehorse
    s = pd.Series(['Whitehorse '])
    trivialpredtrain = s.repeat([len(y_train_final)])
    trivialpredtest = s.repeat([len(y_test_final)])
    trivialpredfull = s.repeat([len(y_full_final)])
    
    trivialtrain_report = classification_report(y_train_final, trivialpredtrain, output_dict=True)
    trivialtest_report = classification_report(y_test_final, trivialpredtest, output_dict=True)
    trivialfull_report = classification_report(y_full_final, trivialpredfull, output_dict=True)
    
    
    
    # Save F1 Data
    lr_dict_train = pd.DataFrame.from_dict(lr_f1_train_report)
    qda_dict_train = pd.DataFrame.from_dict(qda_f1_train_report)
    nn_dict_train = pd.DataFrame.from_dict(nn_f1_train_report)
    rbfsvm_dict_train = pd.DataFrame.from_dict(rbfsvm_f1_train_report)
    nb_dict_train = pd.DataFrame.from_dict(nb_f1_train_report)
    ann_dict_train = pd.DataFrame.from_dict(ann_f1_train_report)
    rf_dict_train = pd.DataFrame.from_dict(rf_f1_train_report)
    ab_dict_train = pd.DataFrame.from_dict(ab_f1_train_report)
    gb_dict_train = pd.DataFrame.from_dict(gb_f1_train_report)
    mcs2a_dict_train = pd.DataFrame.from_dict(mcs2atrain_report)
    mcs2b_dict_train = pd.DataFrame.from_dict(mcs2btrain_report)
    mcs3a_dict_train = pd.DataFrame.from_dict(mcs3atrain_report)
    mcs3b_dict_train = pd.DataFrame.from_dict(mcs3btrain_report)
    mcs5_dict_train = pd.DataFrame.from_dict(mcs5train_report)
    mcs6_dict_train = pd.DataFrame.from_dict(mcs6train_report)
    mcs7_dict_train = pd.DataFrame.from_dict(mcs7train_report)
    mcs8_dict_train = pd.DataFrame.from_dict(mcs8train_report)
    mcs9_dict_train = pd.DataFrame.from_dict(mcs9train_report)
    amcs2a_dict_train = pd.DataFrame.from_dict(alrmcs2atrain_report)
    amcs2b_dict_train = pd.DataFrame.from_dict(alrmcs2btrain_report)
    amcs3a_dict_train = pd.DataFrame.from_dict(alrmcs3atrain_report)
    amcs3b_dict_train = pd.DataFrame.from_dict(alrmcs3btrain_report)
    amcs5_dict_train = pd.DataFrame.from_dict(alrmcs5train_report)
    amcs6_dict_train = pd.DataFrame.from_dict(alrmcs6train_report)
    amcs7_dict_train = pd.DataFrame.from_dict(alrmcs7train_report)
    amcs8_dict_train = pd.DataFrame.from_dict(alrmcs8train_report)
    amcs9_dict_train = pd.DataFrame.from_dict(alrmcs9train_report)
    bmcs2a_dict_train = pd.DataFrame.from_dict(blrmcs2atrain_report)
    bmcs2b_dict_train = pd.DataFrame.from_dict(blrmcs2btrain_report)
    bmcs3a_dict_train = pd.DataFrame.from_dict(blrmcs3atrain_report)
    bmcs3b_dict_train = pd.DataFrame.from_dict(blrmcs3btrain_report)
    bmcs5_dict_train = pd.DataFrame.from_dict(blrmcs5train_report)
    bmcs6_dict_train = pd.DataFrame.from_dict(blrmcs6train_report)
    bmcs7_dict_train = pd.DataFrame.from_dict(blrmcs7train_report)
    bmcs8_dict_train = pd.DataFrame.from_dict(blrmcs8train_report)
    bmcs9_dict_train = pd.DataFrame.from_dict(blrmcs9train_report)
    cmcs2a_dict_train = pd.DataFrame.from_dict(clrmcs2atrain_report)
    cmcs2b_dict_train = pd.DataFrame.from_dict(clrmcs2btrain_report)
    cmcs3a_dict_train = pd.DataFrame.from_dict(clrmcs3atrain_report)
    cmcs3b_dict_train = pd.DataFrame.from_dict(clrmcs3btrain_report)
    cmcs5_dict_train = pd.DataFrame.from_dict(clrmcs5train_report)
    cmcs6_dict_train = pd.DataFrame.from_dict(clrmcs6train_report)
    cmcs7_dict_train = pd.DataFrame.from_dict(clrmcs7train_report)
    cmcs8_dict_train = pd.DataFrame.from_dict(clrmcs8train_report)
    cmcs9_dict_train = pd.DataFrame.from_dict(clrmcs9train_report)
    dmcs2a_dict_train = pd.DataFrame.from_dict(dlrmcs2atrain_report)
    dmcs2b_dict_train = pd.DataFrame.from_dict(dlrmcs2btrain_report)
    dmcs3a_dict_train = pd.DataFrame.from_dict(dlrmcs3atrain_report)
    dmcs3b_dict_train = pd.DataFrame.from_dict(dlrmcs3btrain_report)
    dmcs5_dict_train = pd.DataFrame.from_dict(dlrmcs5train_report)
    dmcs6_dict_train = pd.DataFrame.from_dict(dlrmcs6train_report)
    dmcs7_dict_train = pd.DataFrame.from_dict(dlrmcs7train_report)
    dmcs8_dict_train = pd.DataFrame.from_dict(dlrmcs8train_report)
    dmcs9_dict_train = pd.DataFrame.from_dict(dlrmcs9train_report)
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
    mcs2a_dict_test = pd.DataFrame.from_dict(mcs2atest_report)
    mcs2b_dict_test = pd.DataFrame.from_dict(mcs2btest_report)
    mcs3a_dict_test = pd.DataFrame.from_dict(mcs3atest_report)
    mcs3b_dict_test = pd.DataFrame.from_dict(mcs3btest_report)
    mcs5_dict_test = pd.DataFrame.from_dict(mcs5test_report)
    mcs6_dict_test = pd.DataFrame.from_dict(mcs6test_report)
    mcs7_dict_test = pd.DataFrame.from_dict(mcs7test_report)
    mcs8_dict_test = pd.DataFrame.from_dict(mcs8test_report)
    mcs9_dict_test = pd.DataFrame.from_dict(mcs9test_report)
    amcs2a_dict_test = pd.DataFrame.from_dict(alrmcs2atest_report)
    amcs2b_dict_test = pd.DataFrame.from_dict(alrmcs2btest_report)
    amcs3a_dict_test = pd.DataFrame.from_dict(alrmcs3atest_report)
    amcs3b_dict_test = pd.DataFrame.from_dict(alrmcs3btest_report)
    amcs5_dict_test = pd.DataFrame.from_dict(alrmcs5test_report)
    amcs6_dict_test = pd.DataFrame.from_dict(alrmcs6test_report)
    amcs7_dict_test = pd.DataFrame.from_dict(alrmcs7test_report)
    amcs8_dict_test = pd.DataFrame.from_dict(alrmcs8test_report)
    amcs9_dict_test = pd.DataFrame.from_dict(alrmcs9test_report)
    bmcs2a_dict_test = pd.DataFrame.from_dict(blrmcs2atest_report)
    bmcs2b_dict_test = pd.DataFrame.from_dict(blrmcs2btest_report)
    bmcs3a_dict_test = pd.DataFrame.from_dict(blrmcs3atest_report)
    bmcs3b_dict_test = pd.DataFrame.from_dict(blrmcs3btest_report)
    bmcs5_dict_test = pd.DataFrame.from_dict(blrmcs5test_report)
    bmcs6_dict_test = pd.DataFrame.from_dict(blrmcs6test_report)
    bmcs7_dict_test = pd.DataFrame.from_dict(blrmcs7test_report)
    bmcs8_dict_test = pd.DataFrame.from_dict(blrmcs8test_report)
    bmcs9_dict_test = pd.DataFrame.from_dict(blrmcs9test_report)
    cmcs2a_dict_test = pd.DataFrame.from_dict(clrmcs2atest_report)
    cmcs2b_dict_test = pd.DataFrame.from_dict(clrmcs2btest_report)
    cmcs3a_dict_test = pd.DataFrame.from_dict(clrmcs3atest_report)
    cmcs3b_dict_test = pd.DataFrame.from_dict(clrmcs3btest_report)
    cmcs5_dict_test = pd.DataFrame.from_dict(clrmcs5test_report)
    cmcs6_dict_test = pd.DataFrame.from_dict(clrmcs6test_report)
    cmcs7_dict_test = pd.DataFrame.from_dict(clrmcs7test_report)
    cmcs8_dict_test = pd.DataFrame.from_dict(clrmcs8test_report)
    cmcs9_dict_test = pd.DataFrame.from_dict(clrmcs9test_report)
    dmcs2a_dict_test = pd.DataFrame.from_dict(dlrmcs2atest_report)
    dmcs2b_dict_test = pd.DataFrame.from_dict(dlrmcs2btest_report)
    dmcs3a_dict_test = pd.DataFrame.from_dict(dlrmcs3atest_report)
    dmcs3b_dict_test = pd.DataFrame.from_dict(dlrmcs3btest_report)
    dmcs5_dict_test = pd.DataFrame.from_dict(dlrmcs5test_report)
    dmcs6_dict_test = pd.DataFrame.from_dict(dlrmcs6test_report)
    dmcs7_dict_test = pd.DataFrame.from_dict(dlrmcs7test_report)
    dmcs8_dict_test = pd.DataFrame.from_dict(dlrmcs8test_report)
    dmcs9_dict_test = pd.DataFrame.from_dict(dlrmcs9test_report)
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
    mcs2a_dict_full = pd.DataFrame.from_dict(mcs2afull_report)
    mcs2b_dict_full = pd.DataFrame.from_dict(mcs2bfull_report)
    mcs3a_dict_full = pd.DataFrame.from_dict(mcs3afull_report)
    mcs3b_dict_full = pd.DataFrame.from_dict(mcs3bfull_report)
    mcs5_dict_full = pd.DataFrame.from_dict(mcs5full_report)
    mcs6_dict_full = pd.DataFrame.from_dict(mcs6full_report)
    mcs7_dict_full = pd.DataFrame.from_dict(mcs7full_report)
    mcs8_dict_full = pd.DataFrame.from_dict(mcs8full_report)
    mcs9_dict_full = pd.DataFrame.from_dict(mcs9full_report)
    amcs2a_dict_full = pd.DataFrame.from_dict(alrmcs2afull_report)
    amcs2b_dict_full = pd.DataFrame.from_dict(alrmcs2bfull_report)
    amcs3a_dict_full = pd.DataFrame.from_dict(alrmcs3afull_report)
    amcs3b_dict_full = pd.DataFrame.from_dict(alrmcs3bfull_report)
    amcs5_dict_full = pd.DataFrame.from_dict(alrmcs5full_report)
    amcs6_dict_full = pd.DataFrame.from_dict(alrmcs6full_report)
    amcs7_dict_full = pd.DataFrame.from_dict(alrmcs7full_report)
    amcs8_dict_full = pd.DataFrame.from_dict(alrmcs8full_report)
    amcs9_dict_full = pd.DataFrame.from_dict(alrmcs9full_report)
    bmcs2a_dict_full = pd.DataFrame.from_dict(blrmcs2afull_report)
    bmcs2b_dict_full = pd.DataFrame.from_dict(blrmcs2bfull_report)
    bmcs3a_dict_full = pd.DataFrame.from_dict(blrmcs3afull_report)
    bmcs3b_dict_full = pd.DataFrame.from_dict(blrmcs3bfull_report)
    bmcs5_dict_full = pd.DataFrame.from_dict(blrmcs5full_report)
    bmcs6_dict_full = pd.DataFrame.from_dict(blrmcs6full_report)
    bmcs7_dict_full = pd.DataFrame.from_dict(blrmcs7full_report)
    bmcs8_dict_full = pd.DataFrame.from_dict(blrmcs8full_report)
    bmcs9_dict_full = pd.DataFrame.from_dict(blrmcs9full_report)
    cmcs2a_dict_full = pd.DataFrame.from_dict(clrmcs2afull_report)
    cmcs2b_dict_full = pd.DataFrame.from_dict(clrmcs2bfull_report)
    cmcs3a_dict_full = pd.DataFrame.from_dict(clrmcs3afull_report)
    cmcs3b_dict_full = pd.DataFrame.from_dict(clrmcs3bfull_report)
    cmcs5_dict_full = pd.DataFrame.from_dict(clrmcs5full_report)
    cmcs6_dict_full = pd.DataFrame.from_dict(clrmcs6full_report)
    cmcs7_dict_full = pd.DataFrame.from_dict(clrmcs7full_report)
    cmcs8_dict_full = pd.DataFrame.from_dict(clrmcs8full_report)
    cmcs9_dict_full = pd.DataFrame.from_dict(clrmcs9full_report)
    dmcs2a_dict_full = pd.DataFrame.from_dict(dlrmcs2afull_report)
    dmcs2b_dict_full = pd.DataFrame.from_dict(dlrmcs2bfull_report)
    dmcs3a_dict_full = pd.DataFrame.from_dict(dlrmcs3afull_report)
    dmcs3b_dict_full = pd.DataFrame.from_dict(dlrmcs3bfull_report)
    dmcs5_dict_full = pd.DataFrame.from_dict(dlrmcs5full_report)
    dmcs6_dict_full = pd.DataFrame.from_dict(dlrmcs6full_report)
    dmcs7_dict_full = pd.DataFrame.from_dict(dlrmcs7full_report)
    dmcs8_dict_full = pd.DataFrame.from_dict(dlrmcs8full_report)
    dmcs9_dict_full = pd.DataFrame.from_dict(dlrmcs9full_report)
    trivial_dict_full = pd.DataFrame.from_dict(trivialfull_report)

    
    f1_matrix_train = pd.concat([lr_dict_train.transpose()['f1-score'], qda_dict_train.transpose()['f1-score'],
                   nn_dict_train.transpose()['f1-score'], rbfsvm_dict_train.transpose()['f1-score'], 
                   nb_dict_train.transpose()['f1-score'], ann_dict_train.transpose()['f1-score'], 
                   rf_dict_train.transpose()['f1-score'], ab_dict_train.transpose()['f1-score'], 
                   gb_dict_train.transpose()['f1-score'],
                   mcs2a_dict_train.transpose()['f1-score'], mcs2b_dict_train.transpose()['f1-score'],
                   mcs3a_dict_train.transpose()['f1-score'], mcs3b_dict_train.transpose()['f1-score'], 
                   mcs5_dict_train.transpose()['f1-score'], mcs6_dict_train.transpose()['f1-score'],
                   mcs7_dict_train.transpose()['f1-score'], mcs8_dict_train.transpose()['f1-score'],
                   mcs9_dict_train.transpose()['f1-score'],
                   amcs2a_dict_train.transpose()['f1-score'], amcs2b_dict_train.transpose()['f1-score'],
                   amcs3a_dict_train.transpose()['f1-score'], amcs3b_dict_train.transpose()['f1-score'], 
                   amcs5_dict_train.transpose()['f1-score'], amcs6_dict_train.transpose()['f1-score'],
                   amcs7_dict_train.transpose()['f1-score'], amcs8_dict_train.transpose()['f1-score'],
                   amcs9_dict_train.transpose()['f1-score'],
                   bmcs2a_dict_train.transpose()['f1-score'], bmcs2b_dict_train.transpose()['f1-score'],
                   bmcs3a_dict_train.transpose()['f1-score'], bmcs3b_dict_train.transpose()['f1-score'], 
                   bmcs5_dict_train.transpose()['f1-score'], bmcs6_dict_train.transpose()['f1-score'],
                   bmcs7_dict_train.transpose()['f1-score'], bmcs8_dict_train.transpose()['f1-score'],
                   bmcs9_dict_train.transpose()['f1-score'],
                   cmcs2a_dict_train.transpose()['f1-score'], cmcs2b_dict_train.transpose()['f1-score'],
                   cmcs3a_dict_train.transpose()['f1-score'], cmcs3b_dict_train.transpose()['f1-score'], 
                   cmcs5_dict_train.transpose()['f1-score'], cmcs6_dict_train.transpose()['f1-score'],
                   cmcs7_dict_train.transpose()['f1-score'], cmcs8_dict_train.transpose()['f1-score'],
                   cmcs9_dict_train.transpose()['f1-score'],
                   dmcs2a_dict_train.transpose()['f1-score'], dmcs2b_dict_train.transpose()['f1-score'],
                   dmcs3a_dict_train.transpose()['f1-score'], dmcs3b_dict_train.transpose()['f1-score'], 
                   dmcs5_dict_train.transpose()['f1-score'], dmcs6_dict_train.transpose()['f1-score'],
                   dmcs7_dict_train.transpose()['f1-score'], dmcs8_dict_train.transpose()['f1-score'],
                   dmcs9_dict_train.transpose()['f1-score'],
                   trivial_dict_train.transpose()['f1-score']], axis=1)

    f1_matrix_test = pd.concat([lr_dict_test.transpose()['f1-score'], qda_dict_test.transpose()['f1-score'],
                   nn_dict_test.transpose()['f1-score'], rbfsvm_dict_test.transpose()['f1-score'], 
                   nb_dict_test.transpose()['f1-score'], ann_dict_test.transpose()['f1-score'], 
                   rf_dict_test.transpose()['f1-score'], ab_dict_test.transpose()['f1-score'], 
                   gb_dict_test.transpose()['f1-score'],
                   mcs2a_dict_test.transpose()['f1-score'], mcs2b_dict_test.transpose()['f1-score'],
                   mcs3a_dict_test.transpose()['f1-score'], mcs3b_dict_test.transpose()['f1-score'], 
                   mcs5_dict_test.transpose()['f1-score'], mcs6_dict_test.transpose()['f1-score'],
                   mcs7_dict_test.transpose()['f1-score'], mcs8_dict_test.transpose()['f1-score'],
                   mcs9_dict_test.transpose()['f1-score'],
                   amcs2a_dict_test.transpose()['f1-score'], amcs2b_dict_test.transpose()['f1-score'],
                   amcs3a_dict_test.transpose()['f1-score'], amcs3b_dict_test.transpose()['f1-score'], 
                   amcs5_dict_test.transpose()['f1-score'], amcs6_dict_test.transpose()['f1-score'],
                   amcs7_dict_test.transpose()['f1-score'], amcs8_dict_test.transpose()['f1-score'],
                   amcs9_dict_test.transpose()['f1-score'],
                   bmcs2a_dict_test.transpose()['f1-score'], bmcs2b_dict_test.transpose()['f1-score'],
                   bmcs3a_dict_test.transpose()['f1-score'], bmcs3b_dict_test.transpose()['f1-score'], 
                   bmcs5_dict_test.transpose()['f1-score'], bmcs6_dict_test.transpose()['f1-score'],
                   bmcs7_dict_test.transpose()['f1-score'], bmcs8_dict_test.transpose()['f1-score'],
                   bmcs9_dict_test.transpose()['f1-score'],
                   cmcs2a_dict_test.transpose()['f1-score'], cmcs2b_dict_test.transpose()['f1-score'],
                   cmcs3a_dict_test.transpose()['f1-score'], cmcs3b_dict_test.transpose()['f1-score'], 
                   cmcs5_dict_test.transpose()['f1-score'], cmcs6_dict_test.transpose()['f1-score'],
                   cmcs7_dict_test.transpose()['f1-score'], cmcs8_dict_test.transpose()['f1-score'],
                   cmcs9_dict_test.transpose()['f1-score'],
                   dmcs2a_dict_test.transpose()['f1-score'], dmcs2b_dict_test.transpose()['f1-score'],
                   dmcs3a_dict_test.transpose()['f1-score'], dmcs3b_dict_test.transpose()['f1-score'], 
                   dmcs5_dict_test.transpose()['f1-score'], dmcs6_dict_test.transpose()['f1-score'],
                   dmcs7_dict_test.transpose()['f1-score'], dmcs8_dict_test.transpose()['f1-score'],
                   dmcs9_dict_test.transpose()['f1-score'],
                   trivial_dict_test.transpose()['f1-score']], axis=1)
    
    f1_matrix_full = pd.concat([lr_dict_full.transpose()['f1-score'], qda_dict_full.transpose()['f1-score'],
                   nn_dict_full.transpose()['f1-score'], rbfsvm_dict_full.transpose()['f1-score'], 
                   nb_dict_full.transpose()['f1-score'], ann_dict_full.transpose()['f1-score'], 
                   rf_dict_full.transpose()['f1-score'], ab_dict_full.transpose()['f1-score'], 
                   gb_dict_full.transpose()['f1-score'],
                   mcs2a_dict_full.transpose()['f1-score'], mcs2b_dict_full.transpose()['f1-score'],
                   mcs3a_dict_full.transpose()['f1-score'], mcs3b_dict_full.transpose()['f1-score'], 
                   mcs5_dict_full.transpose()['f1-score'], mcs6_dict_full.transpose()['f1-score'],
                   mcs7_dict_full.transpose()['f1-score'], mcs8_dict_full.transpose()['f1-score'],
                   mcs9_dict_full.transpose()['f1-score'],
                   amcs2a_dict_full.transpose()['f1-score'], amcs2b_dict_full.transpose()['f1-score'],
                   amcs3a_dict_full.transpose()['f1-score'], amcs3b_dict_full.transpose()['f1-score'], 
                   amcs5_dict_full.transpose()['f1-score'], amcs6_dict_full.transpose()['f1-score'],
                   amcs7_dict_full.transpose()['f1-score'], amcs8_dict_full.transpose()['f1-score'],
                   amcs9_dict_full.transpose()['f1-score'],
                   bmcs2a_dict_full.transpose()['f1-score'], bmcs2b_dict_full.transpose()['f1-score'],
                   bmcs3a_dict_full.transpose()['f1-score'], bmcs3b_dict_full.transpose()['f1-score'], 
                   bmcs5_dict_full.transpose()['f1-score'], bmcs6_dict_full.transpose()['f1-score'],
                   bmcs7_dict_full.transpose()['f1-score'], bmcs8_dict_full.transpose()['f1-score'],
                   bmcs9_dict_full.transpose()['f1-score'],
                   cmcs2a_dict_full.transpose()['f1-score'], cmcs2b_dict_full.transpose()['f1-score'],
                   cmcs3a_dict_full.transpose()['f1-score'], cmcs3b_dict_full.transpose()['f1-score'], 
                   cmcs5_dict_full.transpose()['f1-score'], cmcs6_dict_full.transpose()['f1-score'],
                   cmcs7_dict_full.transpose()['f1-score'], cmcs8_dict_full.transpose()['f1-score'],
                   cmcs9_dict_full.transpose()['f1-score'],
                   dmcs2a_dict_full.transpose()['f1-score'], dmcs2b_dict_full.transpose()['f1-score'],
                   dmcs3a_dict_full.transpose()['f1-score'], dmcs3b_dict_full.transpose()['f1-score'], 
                   dmcs5_dict_full.transpose()['f1-score'], dmcs6_dict_full.transpose()['f1-score'],
                   dmcs7_dict_full.transpose()['f1-score'], dmcs8_dict_full.transpose()['f1-score'],
                   dmcs9_dict_full.transpose()['f1-score'],
                   trivial_dict_full.transpose()['f1-score']], axis=1)

    summary_f1_macros_train = pd.concat([summary_f1_macros_train, f1_matrix_train.iloc[[6]]], axis = 0)
    summary_f1_macros_test = pd.concat([summary_f1_macros_test, f1_matrix_test.iloc[[6]]], axis = 0)
    summary_f1_macros_full = pd.concat([summary_f1_macros_full, f1_matrix_full.iloc[[6]]], axis = 0)
    
    summary_f1_averages_train = pd.concat([summary_f1_averages_train, f1_matrix_train.iloc[[5]]], axis = 0)
    summary_f1_averages_test = pd.concat([summary_f1_averages_test, f1_matrix_test.iloc[[5]]], axis = 0)
    summary_f1_averages_full = pd.concat([summary_f1_averages_full, f1_matrix_full.iloc[[5]]], axis = 0)

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
    
    dfcv.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7dfcvseed'+str(seed)+'.csv', index=None, header = True)
    f1_matrix_train.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7f1trainseed'+str(seed)+'.csv', index=None, header = True)
    f1_matrix_test.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7f1testseed'+str(seed)+'.csv', index=None, header = True)
    f1_matrix_full.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7f1fullseed'+str(seed)+'.csv', index=None, header = True)
    dfbp.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7dfbpdataseed'+str(seed)+'.csv', index=None, header = True)

    summary_f1_macros_train.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7macrosummarytrainseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_macros_test.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7macrosummarytestseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_macros_full.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7macrosummaryfullseed'+str(seed)+'.csv', index=None, header = True)

    summary_f1_averages_train.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7averagesummarytrainseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_averages_test.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7averagesummarytestseed'+str(seed)+'.csv', index=None, header = True)
    summary_f1_averages_full.to_csv(r'C:\Users\Timothy\Dropbox\GenFive Programming Files\ComplexMCSResults\v1g7averagesummaryfullseed'+str(seed)+'.csv', index=None, header = True)
