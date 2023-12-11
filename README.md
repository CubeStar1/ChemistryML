# ChemPredictor

## Overview

ChemPredictor aims to analyze Molecular Properties of known compounds and construct an advanced Artificial Neural Network (ANN) model capable of accurately predicting these properties for unknown compounds. This project seamlessly integrates the principles of chemistry, mathematics, and Python programming to develop and deploy an ANN model for Molecular Property Prediction (MPP).

### Objectives

1. **Molecular Property Analysis:** Explore the Molecular Properties of known compounds.
2. **ANN Model Construction:** Build a robust Artificial Neural Network model for accurate prediction of molecular properties.
3. **Interdisciplinary Approach:** Integrate concepts from chemistry, mathematics, and programming to enhance the model's effectiveness.
4. **User-Friendly Interface:** Develop an interactive [WebUI using Streamlit](https://chemistryml-v2.streamlit.app/) for seamless and user-friendly predictions.
5. **Targeted Properties:** Focus on predicting a set of 9 molecular properties crucial for comprehensive chemical analysis:

   **Table 1** - Predicted properties of the QM9 dataset
   
   | No. | Property | Unit      | Description                             |
   |-----|----------|-----------|-----------------------------------------|
   | 1   | μ        | D         | Dipole moment                           |
   | 2   | α        | a³        | Isotropic polarizability                |
   | 3   | homo     | Ha        | Energy of HOMO                          |
   | 4   | lumo     | Ha        | Energy of LUMO                          |
   | 5   | gap      | Ha        | Gap (lumo − homo)                       |
   | 6   | U        | Ha        | Internal energy at 298.15 K             |
   | 7   | H        | Ha        | Enthalpy at 298.15 K                    |
   | 8   | G        | Ha        | Free energy at 298.15 K                 |
   | 9   | Cv       | cal/mol K | Heat capacity at 298.15 K               |

    - **Dipole moment (µ):** Measurement of polarity of a molecule.
    - **Electronic polarizability (α):** Tendency of non-polar molecules to shift their electron clouds relative to their nuclei.
    - **Energy of HOMO:** Energy of the highest occupied Molecular Orbital.
    - **Energy of LUMO:** Energy of the lowest unoccupied Molecular Orbital.
    - **Band Gap Energy:** Energy of LUMO – Energy of HOMO.
    - **Internal energy of atomization (U):** Energy required to break a molecule into separate atoms.
    - **Enthalpy of atomization (H):** Amount of enthalpy change when a compound's bonds are broken, and the component atoms are separated into single atoms.
    - **Free energy of atomization (G):** Extra energy needed to break up a molecule into separate atoms.
    - **Heat capacity (Cv):** Amount of heat required to increase the temperature of the molecule by one degree.

### Key Features

- **QM9 Dataset:** Utilize the QM9 dataset—a benchmark in quantum chemistry—for training and testing the model.
- **Libraries Used:** Leverage TensorFlow, scikit-learn, and RDKit for efficient machine learning and chemical informatics.
- **WebUI:** Create an intuitive [WebUI using Streamlit](https://chemistryml-v2.streamlit.app/) to facilitate easy predictions and exploration of results.




## Usage


### Installation and Setup

To run the ChemPredictor WebUI, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CubeStar1/ChemistryML.git
   cd ChemPredictor
2. **Create a virtual environment:**

   ```bash
   python -m venv venv
3. **Activate the virtual environment:**

- On Windows:   
   ```bash
  .\venv\Scripts\activate
- On Unix or MacOS:
   ```bash
  source venv/bin/activate
4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   
### How to Use
1. **Run the Streamlit app:**

   ```bash
    streamlit run WebUIv2.py
2. **Open the WebUI in your browser:** 

- Open your web browser and navigate to http://localhost:8501 to use the ChemPredictor WebUI.

3. **Predict Molecular Properties:**

   Choose one of the following ways to predict molecular properties:

   - Enter SMILES String:

     - Enter the SMILES string of the compound in the provided input field.
   - Upload SMILES CSV File:

      - Upload a CSV file containing a set of SMILES strings for bulk prediction.
   - Draw Molecule:

     - Utilize the interactive drawing board to draw the molecule for prediction.
4. **Click on the Predict Button:**
- Once the input is provided (SMILES string, CSV file, or drawn molecule), click on the "Predict" button.
5. **View Predicted Values:**
- The predicted values will be displayed on the right side of the screen.

## Hosted Version
For a quick demo, you can also access the hosted version of ChemPredictor at https://chemistryml-v2.streamlit.app/