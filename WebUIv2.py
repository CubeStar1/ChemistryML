
# Libraries
# ---------------------------------------------

# Streamlit - To build the web application
import streamlit as st
from streamlit_option_menu import option_menu
# ---------------------------------------------

# Data manipulation
import pandas as pd   # To read the dataset
import joblib # To load the scaler
import numpy as np # To manipulate the data
# ---------------------------------------------

# Molecular Descriptors
from rdkit import Chem  # To extract information of the molecules
from rdkit.Chem import Draw  # To draw the molecules
from mordred import Calculator, descriptors  # To calculate descriptors
# -----------------------------------------------------------------------

# Sklearn - To split the dataset
from sklearn.model_selection import train_test_split, KFold # To split the dataset
# ----------------------------------------------------------------------------------

# Artificial Neural Network
from tensorflow import keras # To build the neural network
from keras.models import Sequential # Type of neural network
from keras.layers import Dense, Dropout # Type of layers
from keras.callbacks import ReduceLROnPlateau # To reduce the learning rate when the model stops improving
# -------------------------------------------------------------------------------------------------------------

# Other - Performance Metrics
import time




from typing import Any, Dict

from sklearn.preprocessing import MinMaxScaler


original_minmax_setstate = MinMaxScaler.__setstate__

def __monkey_patch_minmax_setstate__(self, state: Dict[str, Any]) -> None:
    state.setdefault("clip", False)
    original_minmax_setstate(self, state)

MinMaxScaler.__setstate__ = __monkey_patch_minmax_setstate__


scaler = joblib.load('ann_scaler_cv.joblib')

# desc_df = pd.read_csv('descriptors_cv.csv')
# desc_df = desc_df.select_dtypes(include=np.number).astype('float32')
# desc_df = desc_df.loc[:, desc_df.var() > 0.0]
# desc_df_columns = desc_df.columns[1:1070]

desc_df = pd.read_csv('desc_names.csv')
desc_df_columns = desc_df['Descriptors'].tolist()

# Defining functions
# ---------------------------------------------

# Function to display molecules
def display_molecule(molecule): # Function to display molecules
    img = Draw.MolToImage(molecule, size=(1000, 1000), fitImage=True)
    st.image(img)

# Function to canonize molecules
def canonize(mol):
    return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

# Function to calculate descriptors
def mordred_descriptors(smiles):
    df = pd.DataFrame(smiles, columns=['SMILES'])
    canonsmiles = [canonize(str(i)) for i in smiles]
    mols = [Chem.MolFromSmiles(str(i)) for i in canonsmiles]
    df['mol'] = mols
    calc = Calculator(descriptors, ignore_3D=True)
    df_desc = calc.pandas(df['mol'])
    df_desc = df_desc.select_dtypes(include=np.number).astype('float32')
    df_desc['ABC'] = [0]
    df_desc['ABCGG'] = [0]
    df_desc = df_desc[desc_df_columns]
    df_descN = pd.DataFrame(scaler.transform(df_desc), columns=df_desc.columns)
    print(df_descN)
    return df_descN

# Function to predict the property of the molecules
def predict_property_cv(X_test_scaled, model):
    X_Cv = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_Cv, columns =['Predicted Cv (J/mol.K)'])
    return predicted


# Defining which property to predict
properties = ['Dipole moment', 'Heat capacity at 298.15 K']

# Defining the layout of the web application
# TITLE
st.set_page_config(page_title="Molecular Properties Prediction App", layout='wide', page_icon=":bar_chart:")
st.title('Molecular Properties Prediction App')


# SIDEBAR
st.sidebar.image('logo_team10alt-modified.png', use_column_width=True)
st.sidebar.title('Are you ready to predict the properties of your molecules?')
st.sidebar.markdown("""---""")
st.sidebar.markdown("## Select property")
property_selection = st.sidebar.multiselect("", options=properties, default=properties[1])
st.sidebar.markdown("""---""")

# Model Selection
st.sidebar.markdown("## Select a model")
model_selection = st.sidebar.selectbox("",
                          ('Artificial Neural Network', 'Random Forest', 'Support Vector Machine'))
st.sidebar.markdown("""---""")
# Input Selection
st.sidebar.markdown("## Select input type")
input_selection = st.sidebar.selectbox("",
                          ('One SMILES input', 'Upload SMILES as file input'))
# 1. One SMILES input
if input_selection == 'One SMILES input':
    smiles_input = st.sidebar.text_input('Please input SMILE strings of the molecules in the box below:',
                                          "")
    prediction = st.sidebar.button('Predict property of molecule(s)', use_container_width=True)
# 2. Upload SMILES as file input
if input_selection == 'Upload SMILES as file input':
    many_SMILES = st.sidebar.file_uploader('or upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in \'SMILES\' column:')
    st.sidebar.markdown (f"""Once you upload your CSV file, click the button below
    to get the {property_selection[0]} prediction """)
    prediction = st.sidebar.button(f'Predict {property_selection[0]} of molecules')


# MAIN PAGE
# st.header(f'{property_selection[0]} Prediction using an Artificial Neural Network', anchor='center', divider='red')
# st.markdown(""" """)
# st.sidebar.markdown("""---""")


model_cv = keras.models.load_model('ann_cv_model.h5')

# PREDICTION PAGE
def Prediction():
    if input_selection == 'One SMILES input' and prediction:
        start_time = time.time()
        progress_bar = st.progress(0, "Predicting...")
        col1, col2, col3= st.columns(3, gap='large')
        with col1:
            st.info('Input Molecule', icon='👇')
            display_molecule(Chem.MolFromSmiles(smiles_input))
            progress_bar.progress(50, "Predicting...")

        df_original = pd.DataFrame ([smiles_input], columns =['SMILES'])
        X_test_scaled = mordred_descriptors([smiles_input])
        X_Cv = predict_property_cv(X_test_scaled, model_cv)
        X_Cv_value = X_Cv['Predicted Cv (J/mol.K)'][0]
        X_Cv_value_rounded = round(X_Cv_value, 2)
        with col2:
            st.info('Predicted Value', icon='📈')
            st.metric(label='Cv(kJ/mol.K)', value= X_Cv_value_rounded)
            time.sleep(1)
            progress_bar.progress(100, "Completed!")
            progress_bar.empty()
        with col3:
            st.info('Time Elapsed', icon='⏱️')
            st.metric(label='Time (s)', value= round(time.time() - start_time, 2))


def page_selection():
    selected = option_menu(
        menu_title="",
        options=["Project Overview", "Predictions", "About"],
        icons=["🏠", "🔮", "📖"],
        menu_icon="🏠",
        default_index=0,
        orientation="horizontal"
    )
    if selected == "Project Overview":
        st.markdown("""---""")
        st.header('Project Overview', anchor='center')

    if selected == "Predictions":
        Prediction()

page_selection()