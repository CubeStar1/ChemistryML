from distutils.command.upload import upload
import numpy as np
import pandas as pd
from matplotlib import image, pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
from mordred import descriptors
# from rdkit.Chem.Draw import MolsToGridImage
# Sklearn
from sklearn.metrics import mean_squared_error
import pickle
# RDKIt
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import rdMolDescriptors
# from rdkit.ML.Descriptors import MoleculeDescriptors
# streamlit ---
import streamlit as st
#from streamlit_gallery import apps, components
#from streamlit_gallery.utils.page import page_group
from PIL import Image
import base64
import io

# Header
st.set_page_config(page_title="Molecular Properties Prediction App", layout='wide')

# User Input
# 1. One or few SMILES input


# User Input
# 1. One or few SMILES input
st.sidebar.title('Are you ready to predict the properties of your molecules?')
st.sidebar.markdown("""---""")


input_type = st.sidebar.selectbox("Select a property",
                          ('Aqueous Solubility', 'Dipole moment', 'Heat capacity at 298.15 K'))
st.sidebar.markdown("""---""")

st.header(f'{input_type} Prediction using an Artificial Neural Network', anchor='center', divider='red')
st.markdown(""" """)
#st.sidebar.markdown("## Please input SMILE strings of the molecules in the box below:")
smiles_list = st.sidebar.text_input('Please input SMILE strings of the molecules in the box below:',
                                          "['CCCCO']")
st.sidebar.markdown("""---""")

# 2. upload many SMILES input
many_SMILES = st.sidebar.file_uploader('or upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in \'SMILES\' column:')
st.sidebar.markdown ("""Once you upload your CSV file, click the button below
to get the solubility prediction """)
prediction = st.sidebar.button('Predict property of molecules')

def display_molecule(molecule):
    img = Draw.MolToImage(molecule)
    st.image(img, use_column_width=True)


if smiles_list != "['CCCCO']":


    mol = Chem.MolFromSmiles(str(smiles_list))

    _discard = AllChem.Compute2DCoords(mol)
    Draw.MolToFile(mol, 'img1.png', size=(300, 300), fitImage=False, imageType='png')

    col1, col2, col3 = st.columns(3)

    df = pd.DataFrame(eval (smiles_list), columns =['SMILES'])

    #========= function call to calculate 200 molecular descriptors using SMILES
    with col1:
        st.image('img1.png')
        # st.image(MolsToGridImage(molslit, molsPerRow=5, returnPNG=True))
        st.markdown(f'### Your input SMILES is: {smiles_list}')

    with col2:
        st.write(' ')

    with col3:
        st.write(' ')



