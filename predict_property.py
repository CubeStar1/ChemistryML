import pandas as pd
from utils import mordred_descriptors_dataframe


def predict_property_cv(X_test_scaled, model):
    X_Cv = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_Cv, columns =['Predicted Cv (cal/mol.K)'])
    return predicted
def predict_property_G(X_test_scaled, model):
    X_G = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_G, columns =['Predicted G (Ha)'])
    return predicted
def predict_property_mu(X_test_scaled, model):
    X_mu = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_mu, columns =['Predicted mu (D)'])
    return predicted


def generate_prediction_dataframe(df, desc_df_columns, scaler, model_cv, model_G, model_mu):
    X_test_scaled = mordred_descriptors_dataframe(df, desc_df_columns, scaler)
    X_Cv = predict_property_cv(X_test_scaled, model_cv)
    X_Cv['Predicted Cv (cal/mol.K)'] = X_Cv['Predicted Cv (cal/mol.K)'].apply(lambda x: round(x, 2))
    X_G = predict_property_G(X_test_scaled, model_G)
    X_G['Predicted G (Ha)'] = X_G['Predicted G (Ha)'].apply(lambda x: round(x, 2))
    X_mu = predict_property_mu(X_test_scaled, model_mu)
    X_mu['Predicted mu (D)'] = X_mu['Predicted mu (D)'].apply(lambda x: round(x, 2))
    output_df = pd.concat([df, X_Cv, X_G, X_mu], axis=1)
    output_df.drop(columns=['mol'], inplace=True)
    return output_df