import pandas as pd
from sklearn import preprocessing

def loadTrainSet():
	train = pd.read_csv('train.csv');
	
	labels = train.target.values;
	train = train.drop('id', axis=1);
	train = train.drop('target', axis=1);
	
	lbl_enc = preprocessing.LabelEncoder();
	labels = lbl_enc.fit_transform(labels);
	
	return (train,labels)
	
def loadTestSet():
	test = pd.read_csv('test.csv');
	test = test.drop('id', axis=1);
	return test;

def saveResult(preds,filename):
	sample = pd.read_csv('sampleSubmission_o.csv')
	preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
	preds.to_csv(filename, index_label='id')
	return;

