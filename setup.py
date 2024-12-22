# To deploy the application as an entire package, we need to create a setup.py file. 
# This file contains the metadata about the package, such as the name, version, and dependencies.
# The setup.py file is used by the setuptools package to install the package and its dependencies.
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .' # To avoid the error in the requirements.txt file

def get_requirements(file_path:str)->List[str]:
    '''
    Read the requirements.txt file and return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements =[req.replace('\n','') for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements
    

setup(
    name = 'mlproject',
    version='0.1',
    author = 'Punith',
    author_email= 'punithsujatha@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')                 #['pandas','numpy','scikit-learn','seaborn','matplotlib']
)