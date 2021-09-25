import numpy as np
import sys


from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import DataStructs

from sklearn.metrics.pairwise import cosine_similarity

##### Level 1 function: #####
def read_sdf_mol(file):
    ### output molecular objects from sdf 
    return [mol for mol in Chem.SDMolSupplier(file) if mol is not None]



def molsfeaturizer(mols):
    ### featurize moleculars using MoganFingerprint 
    fps = []
    for mol in mols:
        arr = np.zeros((0,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fps = np.array(fps)
    # print (fps)
    # sys.exit(0)
    return fps





##### Level 2 functions:  ####
def get_cosin_similarity(sdf1,sdf2,feature): 
    ## feature: what feature to use for similarity measure 
    data_1 = read_sdf_mol(sdf1)
    data_2 = read_sdf_mol(sdf2)
    if feature == "Morgan":
        data_1_features = molsfeaturizer(data_1)
        data_2_features = molsfeaturizer(data_2)
    else: 
        sys.exit("No such featurize method")
    cos_sim=cosine_similarity(data_1_features,data_2_features)
    return cos_sim
