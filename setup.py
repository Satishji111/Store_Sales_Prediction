from setuptools import find_packages, setup
from typing import List


hyphen_e_dot='-e .'
def get_requirements(file_path:str)-> List[str]:
    '''
    This function will return list of requirements.'''
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    return requirements

setup( 
name='store_sales_prediction',
author='Satish',
author_email='satish.data.analyst111@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)