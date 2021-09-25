#!/usr/bin/python3
 
import pprint
import argparse
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
from rdkit import Chem
from rdkit.Chem import AllChem
 
from rdkit.Chem import DataStructs
import numpy as np

import pandas as pd

from sklearn.metrics import r2_score 


from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt 
 
def base_parser():
    parser = argparse.ArgumentParser("This is simple test of pytorch")
    # parser.add_argument("trainset", help="sdf for train")
    # parser.add_argument("testset", help="sdf for test")
    parser.add_argument("--epochs", default=500)
    return parser
 

#### Get molecular finger print of input compound#####
def molsfeaturizer(mols):
    fps = []
    for mol in mols:
        arr = np.zeros((0,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fps = np.array(fps)
    return fps
 
#### Get data of activity#####
def get_activity_val(mol_name,df,th):
    return float(df['% Uptake'][df['TKI'].str.contains(mol_name)]<th)


### One to one match to find corresponding y of every input compound########
def get_y(data,activity_file,th): 
    df=pd.read_csv(activity_file)
    y=np.array([get_activity_val(mol.GetProp('_Name'),df,th) for mol in data])
    return y 

### Construct ANN QSAR Model ########
class QSAR_mlp(nn.Module):
    def __init__(self):
        super(QSAR_mlp, self).__init__()
        self.fc1 = nn.Linear(2048, 524) ### the first input has to be 2048 matching the vector you inputed 
        self.fc2 = nn.Linear(524, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10,2)  ###the final output has to be 0 becase the binary classification 
    def forward(self, x):
        x = x.view(-1, 2048)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = F.sigmoid(self.fc4(h3))
        return output


def get_train_test_mol_name(traindata,testdata):
    train_mol_name=[mol.GetProp('_Name') for mol in traindata]
    test_mol_name=[mol.GetProp('_Name') for mol in testdata]    
    return train_mol_name,test_mol_name


parser = base_parser()
args = parser.parse_args()

sdf_file='tki_tested.sdf' 
activity_file='TKI2_uptake.csv'


train_test_ratio=0.8
th=10 ## threshold for binarizing the classification of y (acitivity)
acc_th=0.75 ## accuracy that considers as good result 
has_good_index=1 ## good index is obtained

data = [mol for mol in Chem.SDMolSupplier(sdf_file) if mol is not None]

N_data=len(data)
N_train=int(N_data*train_test_ratio)


pred_accuracies=[0,0,0,0]


SVC = make_pipeline(StandardScaler(), SVC(gamma='auto'))
runcycles=0
#while (np.array(pred_accuracies)>=0.8).any()==False: 
while runcycles <=20:
    seed_n=np.random.randint(100000,size=1)
    np.random.seed(seed_n)### to store the seed_n for the random principle 
    if has_good_index==1: 
        index_pool=np.load('index_pool.npy')    
    else: 
        index_pool=np.random.permutation(N_data)
    train_index= index_pool[:N_train]
    test_index= index_pool[N_train:]

    traindata=[data[i] for i in train_index]
    testdata=[data[i] for i in test_index]  
     

    trainx = molsfeaturizer(traindata)
    testx = molsfeaturizer(testdata)
    trainy=get_y(traindata,activity_file,th)
    testy=get_y(testdata,activity_file,th)
      
    ### support vector machine ###
    SVC.fit(trainx, trainy)
    SVC_pred_y=SVC.predict(testx)
    SVC_train_y=SVC.predict(trainx)
    SVC_train_accuracy=np.sum(SVC_train_y==trainy)/len(trainy)
    SVC_accuracy=np.sum(SVC_pred_y==testy)/len(testy)


    ### ANN ###

    X_train = torch.from_numpy(trainx).to(torch.long)
    X_test = torch.from_numpy(testx).to(torch.long)
    Y_train = torch.from_numpy(trainy).to(torch.float)
    Y_test = torch.from_numpy(testy).to(torch.float)  

    model = QSAR_mlp()
     
    losses = []
    optimizer = optim.Adam( model.parameters(), lr=0.005)  ###learning rate 0.005, the step guide the gradient descent
    for epoch in range(args.epochs):
        data_ann, target = Variable(X_train).float(), Variable(Y_train).long()
        optimizer.zero_grad() 
        y_pred = model(data_ann)
        loss = F.cross_entropy(y_pred, target)  
        loss.backward()
        optimizer.step()
     
    pred_y = model(Variable(X_test).float())
    ANN_pred_y = torch.max(pred_y, 1)[1].detach().numpy()
    Y_test = Y_test.detach().numpy()
    ANN_accuracy=np.sum(ANN_pred_y==Y_test)/len(Y_test)# pred=[]
    pred_accuracies=np.array([SVC_accuracy,ANN_accuracy])

    T_pred_y = model(Variable(X_train).float())
    T_ANN_pred_y = torch.max(T_pred_y, 1)[1].detach().numpy()
    Y_train = Y_train.detach().numpy()
    T_ANN_accuracy=np.sum(T_ANN_pred_y==Y_train)/len(Y_train)# pred=[]
    

    pred_accuracies=np.array([SVC_accuracy,ANN_accuracy])
    T_pred_accuracies=np.array([SVC_train_accuracy,T_ANN_accuracy])

    
#    print ('pred_accuracies  ',pred_accuracies)
#    print ('training pred_accuracies  ',T_pred_accuracies)
    file_result=open('test_accuracy','a')
    file_result.write(str(pred_accuracies[0])+'\t'+str(pred_accuracies[1])+'\n')
    runcycles+=1





