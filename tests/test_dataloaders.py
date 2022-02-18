import sys

sys.path.insert(1, './src/')
from utils import split_data_into_dataloaders


X = [1,2,3,4]
y = [0,1,0,1]



def test_split_data_into_dataloaders():
	split_data_into_dataloaders(X,y,0.7,0.4)
