# Libraries
# ---------------------------------------------
# General
import os
import glob
from typing import Any, Dict
# Data manipulation
import pandas as pd   # To read the dataset
import joblib # To load the scaler
import numpy as np # To manipulate the data
# Streamlit - To build the web application
import streamlit as st
from streamlit_option_menu import option_menu
# Ketcher - To draw molecules
from streamlit_ketcher import st_ketcher
# 3Dmol - To display molecules
from utils import display_3D_molecule
from stmol import showmol
import py3Dmol
# Utilities
import utils
from utils import display_molecule_in_dataframe_as_html
from predict_property import generate_prediction_dataframe
# RDKit - To handle molecules
from rdkit import Chem  # To extract information of the molecules
from rdkit.Chem import AllChem# To draw the molecules
# Tensorflow - To load the neural network
from tensorflow import keras # To build the neural network
# Sklearn - To load the scaler
from sklearn.preprocessing import MinMaxScaler
# ---------------------------------------------

# Monkey patching the MinMaxScaler to avoid clipping
original_minmax_setstate = MinMaxScaler.__setstate__
def __monkey_patch_minmax_setstate__(self, state: Dict[str, Any]) -> None:
    state.setdefault("clip", False)
    original_minmax_setstate(self, state)
MinMaxScaler.__setstate__ = __monkey_patch_minmax_setstate__

# Loading the descriptor columns
desc_df = pd.read_csv('utilities/descriptors/descriptor_columns_full1.csv')
desc_df_columns = desc_df['descriptor'].tolist()

# Loading the scaler and the models
scaler = joblib.load('utilities/scalers/ann_scaler_cv_full (1).joblib')
model_cv = keras.models.load_model('utilities/models/ann_cv_model_full.h5')
model_G = keras.models.load_model('utilities/models/ann_G_model_full.h5')
model_mu = keras.models.load_model('utilities/models/ann_mu_model_full.h5')

# Defining which property to predict
properties = ['HOMO', 'LUMO', 'Band Gap', 'Polarizability', 'Dipole moment', 'U', 'H', 'G','Cv']

# Defining the layout of the web application
# TITLE
st.set_page_config(page_title="Molecular Properties Prediction App", layout='wide', page_icon=":bar_chart:")
st.title('Molecular Properties Prediction App')
st.markdown("""---""")

# SIDEBAR
#st.sidebar.image('logo_team10alt-modified.png', use_column_width=True)
#st.sidebar.title('Are you ready to predict the properties of your molecules?')
with st.sidebar:
    #logo_smile = "c1ccccc1"
    #display_3D_molecule(logo_smile, width=200, height=200)

    # Property Selection
    st.markdown("## Select property")
    property_selection = st.sidebar.multiselect("", options=properties, default=properties)
    #st.markdown("""---""")
    # Model Selection
    st.markdown("## Select a model")
    model_selection = st.selectbox("",
                              ('Artificial Neural Network', 'Random Forest', 'Support Vector Machine'))
    #st.markdown("""---""")
    # Input Selection
    st.markdown("## Select input type")
    input_selection = st.selectbox("",
                              ('SMILES input', 'Upload SMILES as file input', 'Draw molecule'))
# 1. SMILES input
if input_selection == 'SMILES input':
    smiles_input = st.sidebar.text_input('Please input SMILE strings of the molecules in the box below:',
                                          "")
    prediction = st.sidebar.button('Predict property of molecule(s)', use_container_width=True)
# 2. Upload SMILES as file input
if input_selection == 'Upload SMILES as file input':
    many_SMILES = st.sidebar.file_uploader('Upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in \'SMILES\' column:')
    prediction = st.sidebar.button(f'Predict property of molecules')


# PREDICTION PAGE
def Prediction():


    #st.header('Select Property', anchor='center')
    #property_selection = st.multiselect("", options=properties, default=properties[-1])
    if input_selection == 'SMILES input' and prediction:
        files = glob.glob('images/*.png')
        for f in files:
            os.remove(f)
        smiles_list = smiles_input.split(",")
        df_original = pd.DataFrame (smiles_list, columns =['SMILES'])
        output_df = generate_prediction_dataframe(df_original, desc_df_columns, scaler, model_cv, model_G, model_mu)
        html_df = display_molecule_in_dataframe_as_html(output_df)
        st.markdown(html_df, unsafe_allow_html=True)

    elif input_selection == 'Upload SMILES as file input' and prediction:
        files = glob.glob('images/*.png')
        for f in files:
            os.remove(f)
        df = pd.read_csv(many_SMILES)
        output_df = generate_prediction_dataframe(df, desc_df_columns, scaler, model_cv, model_G, model_mu)
        html_df = display_molecule_in_dataframe_as_html(output_df)
        st.markdown(html_df,unsafe_allow_html=True)


    elif input_selection == 'Draw molecule':
        # famous_molecules = [
        #     ('☕', 'Caffeine'),
        #     ('🥱', 'Melatonin'),
        #     ('🚬', 'Nicotine'),
        #     ('🌨️', 'Cocaine'),
        #     ('💊', 'Aspirin'),
        #     ('🍄', 'Psilocybine'),
        #     ('💎', 'Lysergide')
        # ]
        st.session_state.molfile = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1"))
        # for mol, column in zip(famous_molecules, st.columns(len(famous_molecules))):
        #     with column:
        #         emoji, name = mol
        #
        #         if st.button(f'{emoji} {name}'):
        #             st.session_state.molfile, st.session_state.chembl_id = utils.name_to_molecule(name)

        files = glob.glob('images/*.png')
        for f in files:
            os.remove(f)
        st.markdown("""---""")
        st.markdown("## Please draw your molecule(s) in the box below:")
        editor_column, results_column = st.columns(2)
        similar_smiles = []
        with editor_column:
            smiles = st_ketcher(st.session_state.molfile)


            with results_column:
                with st.expander("Show similar molecules"):
                    similarity_threshold = st.slider("Similarity threshold:", min_value=60, max_value=100)
                    similar_molecules = utils.find_similar_molecules(smiles, similarity_threshold)
                    if not similar_molecules:
                        st.warning("No results found")
                    else:
                        table = utils.render_similarity_table(similar_molecules)
                        similar_smiles = utils.get_similar_smiles(similar_molecules)
                        st.markdown(f'<div id="" style="overflow:scroll; height:400px; padding-left: 20px;">{table}</div>',
                                    unsafe_allow_html=True)

        smile_code = smiles.replace('.',', ')
        molecule = st.sidebar.text_input("SMILES Representation", smile_code)
        moleculesList = molecule.split(",")
        prediction2 = st.sidebar.button('Predict property of molecule(s)')
        with results_column:
            with st.expander("Show 3D molecule(s)"):
                if len(moleculesList) >0 and molecule != "":
                    for  smile in moleculesList:

                        display_3D_molecule(smile, width=400, height=400)

                # mol = Chem.MolFromSmiles(smile_code)
                # mol = Chem.AddHs(mol)
                # AllChem.EmbedMolecule(mol)
                # AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                # mblock = Chem.MolToMolBlock(mol)
                # view = py3Dmol.view(width=400, height=400)
                # view.addModel(mblock, 'mol')
                # view.setStyle({"stick": {}})
                # view.zoomTo()
                # view.spin(spin_on=True)
                # showmol(view)

        if prediction2:

            moleculesList = molecule.split(",")
            df_original = pd.DataFrame(moleculesList, columns=['SMILES'])
            output_df = generate_prediction_dataframe(df_original, desc_df_columns, scaler, model_cv, model_G, model_mu)
            html_df = display_molecule_in_dataframe_as_html(output_df)
            with results_column:
                with st.expander("Show predicted properties", expanded=True):
                    st.markdown(f'<div id="" style="overflow:scroll; height:400px; padding-left: 20px;">{html_df}</div>',
                                    unsafe_allow_html=True)


# AI CHATBOT PAGE
def Chat():
    pass

def page_selection():
    selected = option_menu(
        menu_title="",
        options=["Predictor", "Project Overview", "Chat"],
        icons=["🏠", "🔮", "✨" ],
        menu_icon="🏠",
        default_index=0,
        orientation="horizontal"
    )
    if selected == "Project Overview":
        st.markdown("""---""")
        st.header('Project Overview', anchor='center')

    if selected == "Predictor":
        Prediction()
    if selected == "Chat":
        st.markdown("""---""")
        st.header('Chat', anchor='center')
        Chat()

page_selection()