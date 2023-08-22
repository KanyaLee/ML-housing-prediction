import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('df_cleaned_for_ML_regression.csv') 
    return df

# Function to train model
def train_model(df, model_choice):
    # Preprocessing
    le = LabelEncoder()
    scaler = MinMaxScaler()

    # Encode district
    df['district'] = le.fit_transform(df['district'])

    # Define features and target
    features = ['district', 'year_built', 'hospital', 'Gym', 'Pool', 'Parking', 'Security', 'CCTV', 'Shop', 'Restaurant', 'Sauna']
    X = df[features].copy()
    y = df['price_sqm']

    # Normalize features

    # X[features] = scaler.fit_transform(X[features])
    for feature in features:
        X[feature] = scaler.fit_transform((X[[feature]]))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = model_choice()
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feature_importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importances_df.sort_values('importance', ascending=False, inplace=True)

    return rmse, r2, feature_importances_df
    

def app():
    st.title('Bangkok Condominium Price Prediction')

    df = load_data()

    st.header('Data')
    st.write(df)

    model_choice = st.selectbox('Choose a model', ('RandomForestRegressor', 'GradientBoostingRegressor'))
    model_choice = RandomForestRegressor if model_choice == 'RandomForestRegressor' else GradientBoostingRegressor

    rmse, r2, feature_importances_df = train_model(df, model_choice)

    st.header('Model Performance')
    st.write('RMSE: ', rmse)
    st.write('R2 Score: ', r2)

    st.header('Feature Importances')
    st.write(feature_importances_df)

    st.bar_chart(feature_importances_df.set_index('feature'))


if __name__ == '__main__':
    app()
