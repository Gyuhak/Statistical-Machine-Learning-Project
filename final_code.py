from collections import Counter
import os
import numpy as np
import pickle
import logging
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random 
from scipy.optimize import minimize
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


def load_and_shuffle(dirPath, dimension=1024, random_seed=0):
	class_number = 0
	data = np.zeros(shape=(1,dimension+1))
	for file in os.listdir(dirPath):
		if file.endswith(".txt"):
			data_X = np.loadtxt(dirPath+'/'+file)[0:100]
			data_y = np.ones(100)*class_number
			# print data_X.shape
			# print data_y.shape
			data_class = np.c_[data_X,data_y.T]
			# print data_class.shape
			# print data.shape
			data = np.concatenate((data, data_class), axis=0)
			class_number += 1
	data = np.delete(data, (0), axis=0)

	np.random.seed(random_seed)
	np.random.shuffle(data)
	np.savetxt('data.txt', data)

	y=data[:,-1]
	X=data[:,0:dimension]

	X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.35, random_state=42)
	X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.23, random_state=42)
	return (X_train, X_valid, X_test, y_train, y_valid, y_test)

def load_and_shuffle_style(dirPath, dimension=1024, random_seed=0):
	class_number = 0
	data = np.zeros(shape=(1,dimension+1))
	for file in os.listdir(dirPath):
		if file.endswith(".txt"):
			data_X = np.loadtxt(dirPath+'/'+file)
			data_y = np.ones(data_X.shape[0])*class_number
			print(data_X.shape)
			print(data_y.shape)
			data_class = np.c_[data_X,data_y.T]
			data = np.concatenate((data, data_class), axis=0)
			class_number += 1
	data = np.delete(data, (0), axis=0)
	print(data.shape)

	np.random.seed(random_seed)
	np.random.shuffle(data)
	np.savetxt('data.txt', data)

	y=data[:,-1]
	X=data[:,0:dimension]

	X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.35, random_state=42)
	X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=220, random_state=42)
	return (X_train, X_valid, X_test, y_train, y_valid, y_test)

def print_error_rate(y_predict, y_actual):
	error_num = 0
	index = 0
	for i in range(y_actual.shape[0]):
		if y_predict[i] != y_actual[i]:
			error_num += 1
	error_rate = 1.0*error_num/y_actual.shape[0]
	print(error_rate)
	return error_rate

def run_SGD(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------SGD------------')
	clf_SGD = SGDClassifier(alpha=0.01, average=True, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)
	clf_SGD.fit(X_train,y_train)

	print('validation:')
	y_valid_predict = []
	for row in X_valid:
		y_valid_predict.append(clf_SGD.predict([row])[0])
	print_error_rate(y_valid_predict, y_valid)

	print('test:')
	y_test_predict = []
	for row in X_test:
		y_test_predict.append(clf_SGD.predict([row])[0])
	print_error_rate(y_test_predict, y_test)
	return (y_valid_predict, y_test_predict)
	

def run_SVM(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------SVM------------')
	clf_SVC = svm.SVC(C=1.0, kernel='poly', degree=5, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
	clf_SVC.fit(X_train, y_train) 
	error_num = 0
	index = 0

	print('validation:')
	y_valid_predict = []
	for row in X_valid:
		y_valid_predict.append(clf_SVC.predict([row])[0])
	print_error_rate(y_valid_predict, y_valid)

	print('test:')
	y_test_predict = []
	for row in X_test:
		y_test_predict.append(clf_SVC.predict([row])[0])
	print_error_rate(y_test_predict, y_test)
	return (y_valid_predict, y_test_predict)

def run_KNN(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------KNN------------')
	neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	neigh.fit(X_train, y_train)
	# KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	print('validation:')
	y_valid_predict = []
	for row in X_valid:
		y_valid_predict.append(neigh.predict([row])[0])
	print_error_rate(y_valid_predict, y_valid)

	print('test:')
	y_test_predict = []
	for row in X_test:
		y_test_predict.append(neigh.predict([row])[0])
	print_error_rate(y_test_predict, y_test)
	return (y_valid_predict, y_test_predict)

def run_LDA(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------LDA------------')
	clf_LDA = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.01)
	clf_LDA.fit(X_train,y_train)
	#LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)
	print('validation:')
	y_valid_predict = []
	for row in X_valid:
		y_valid_predict.append(clf_LDA.predict([row])[0])
	print_error_rate(y_valid_predict, y_valid)

	print('test:')
	y_test_predict = []
	for row in X_test:
		y_test_predict.append(clf_LDA.predict([row])[0])
	print_error_rate(y_test_predict, y_test)
	return (y_valid_predict, y_test_predict)

def run_QUA(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------QUA------------')
	clf_QUA = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariances=True, tol=0.01)
	clf_QUA.fit(X_train, y_train)
	# QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariances=False, tol=0.01)
	print('validation:')
	y_valid_predict = []
	for row in X_valid:
		y_valid_predict.append(clf_QUA.predict([row])[0])
	print_error_rate(y_valid_predict, y_valid)

	print('test:')
	y_test_predict = []
	for row in X_test:
		y_test_predict.append(clf_QUA.predict([row])[0])
	print_error_rate(y_test_predict, y_test)
	return (y_valid_predict, y_test_predict)

def run_RF(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------Random Forest------------')
	clf_RF = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=40, min_samples_split=8, 
		min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
		bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=True, class_weight=None)
	clf_RF = clf_RF.fit(X_train, y_train)
	print('validation:')
	y_valid_predict = []
	for row in X_valid:
		y_valid_predict.append(clf_RF.predict([row])[0])
	print_error_rate(y_valid_predict, y_valid)

	print('test:')
	y_test_predict = []
	for row in X_test:
		y_test_predict.append(clf_RF.predict([row])[0])
	print_error_rate(y_test_predict, y_test)
	return (y_valid_predict, y_test_predict)


def SGD_cross(X_train, X_valid, X_test, y_train, y_valid, y_test):
	print('------------SGD Cross Valiedation------------')
	clf_SGD = SGDClassifier()
	X_axis = np.array([10**-10,10**-9,10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5,10**6])
	# X_axis = np.arange(1,100)
	y_axis_valid = np.zeros(X_axis.shape[0])
	y_axis_test = np.zeros(X_axis.shape[0])
	for index in range(0,X_axis.shape[0]):
		SGDClassifier(alpha=X_axis[index], average=True, class_weight=None, epsilon=0.1, eta0=0.0, 
			fit_intercept=True, l1_ratio=0.15,learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, 
			penalty='l1', power_t=0.5, random_state=None, shuffle=True,verbose=0, warm_start=True)
		clf_SGD.fit(X_train,y_train)
		y_valid_predict = []
		print('validation for alpha is', X_axis[index])
		for row in X_valid:
			y_valid_predict.append(clf_SGD.predict([row])[0])
		# print_error_rate(y_valid_predict, y_valid)
		y_axis_valid[index] = print_error_rate(y_valid_predict, y_valid)

		y_test_predict = []
		for row in X_test:
			y_test_predict.append(clf_SGD.predict([row])[0])
		y_axis_test[index] = print_error_rate(y_test_predict, y_test)
	plt.plot(X_axis,y_axis_valid)
	plt.plot(X_axis,y_axis_test)
	plt.xlabel("alpha")
	plt.xscale('log')
	plt.legend(["Validation set", "Test set"])
	plt.ylabel("minimum square loss")
	plt.title("minimum square loss on different alph")
	plt.show()
  #       print 'test:'
  #       y_test_predict = []
        
  #       for row in X_test:
  #       	y_test_predict.append(clf_SGD.predict([row])[0])
		# print_error_rate(y_test_predict, y_test)

def emsemble_by_majority(y_valid_predict_sgd, y_test_predict_sgd, y_valid_predict_svm, y_test_predict_svm,
		y_valid_predict_rf, y_test_predict_rf, y_valid, y_test):
	print('------------Ensemble------------')
	print('validation:')
	y_valid_ensemble = []
	for i in range(y_valid.shape[0]):
		# print i
		c = Counter([y_valid_predict_sgd[i], y_valid_predict_svm[i], y_valid_predict_rf[i]])
		value, count = c.most_common()[0]
		#print value, count
		if count != 0:
			y_valid_ensemble.append(value)
		else:
			y_valid_ensemble.append(y_valid_predict_svm[i])
	print_error_rate(y_valid_ensemble, y_valid)

	print('test:')
	y_test_ensemble = []
	for i in range(y_test.shape[0]):
		#print i
		c = Counter([y_test_predict_sgd[i], y_test_predict_svm[i], y_test_predict_rf[i]])
		value, count = c.most_common()[0]
		#print value, count
		if count != 0:
			y_test_ensemble.append(value)
		else:
			y_test_ensemble.append(y_test_predict_svm[i])
	print_error_rate(y_test_ensemble, y_test)


def main():
	# this dirPath point to the directory contains all data files you want to run.
	# Not neccessary to trim, I use first 100 rows of each file
	dirPath = '/Users/gyuhak/Downloads/mlproject'
	# Use random seed to get different shuffle
	X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_shuffle(dirPath, dimension=1024, random_seed=4)
	# X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_shuffle_style(dirPath, dimension=1024, random_seed=0)
	# print X_test
	# y_valid_predict_sgd, y_test_predict_sgd = run_SGD(X_train, X_valid, X_test, y_train, y_valid, y_test)
	# y_valid_predict_svm, y_test_predict_svm = run_SVM(X_train, X_valid, X_test, y_train, y_valid, y_test)
	# y_valid_predict_knn, y_test_predict_knn = run_KNN(X_train, X_valid, X_test, y_train, y_valid, y_test)
	# y_valid_predict_lda, y_test_predict_lda = run_LDA(X_train, X_valid, X_test, y_train, y_valid, y_test)
	# y_valid_predict_qua, y_test_predict_qua = run_QUA(X_train, X_valid, X_test, y_train, y_valid, y_test)
	y_valid_predict_rf, y_test_predict_rf = run_RF(X_train, X_valid, X_test, y_train, y_valid, y_test)
	# emsemble_by_majority(y_valid_predict_sgd, y_test_predict_sgd, y_valid_predict_svm, y_test_predict_svm,
	# 	y_valid_predict_rf, y_test_predict_rf, y_valid, y_test)


	#alpha=[0.0001,0.001,0.01,0.1,1,10,100]
	#SGD_cross(X_train, X_valid, X_test, y_train, y_valid, y_test)

if __name__ == "__main__":
    main()