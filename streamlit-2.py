import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    df = pd.read_csv('df_cleaned_for_ML_regression.csv') 
    return df.copy()

def train_model(df, model_choice):
    le = LabelEncoder()
    scaler = MinMaxScaler()
    df['district'] = le.fit_transform(df['district'])

    features = ['district', 'year_built', 'hospital', 'Gym', 'Pool', 'Parking', 'Security', 'CCTV', 'Shop', 'Restaurant', 'Sauna']
    X = df[features].copy()
    y = df['price_sqm']

    for feature in features:
        X[feature] = scaler.fit_transform((X[[feature]]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_choice()
    model.fit(X_train, y_train)

    #predicting
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feature_importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importances_df.sort_values('importance', ascending=False, inplace=True)

    return model, rmse, r2, feature_importances_df

def app():
    st.title('Bangkok Condominium Price Prediction - Lynnda')

    df = load_data()

    st.header('Data')
    st.write(df)

    model_choice = st.selectbox('Choose a model', ('RandomForestRegressor', 'GradientBoostingRegressor'))
    model_choice = RandomForestRegressor if model_choice == 'RandomForestRegressor' else GradientBoostingRegressor

    model, rmse, r2, feature_importances_df = train_model(df, model_choice)

    st.header('Model Performance')
    st.write('RMSE: ', rmse)
    st.write('R2 Score: ', r2)

    st.header('Feature Importances')
    st.write(feature_importances_df)

    st.bar_chart(feature_importances_df.set_index('feature'))

    #dropdown
    district_names = df['district'].unique()
    district = st.selectbox('Choose a district', district_names)
    district_encoded = df[df['district'] == district]['district'].values[0]

    features = ['district', 'year_built', 'hospital', 'Gym', 'Pool', 'Parking', 'Security', 'CCTV', 'Shop', 'Restaurant', 'Sauna']
    X_district = pd.DataFrame([[district_encoded] + [0] * (len(features) - 1)], columns=features)
    predicted_price = model.predict(X_district)[0]

    st.header('Predicted Price for Selected District')
    st.write(f'The predicted price per sqm for district {district} in the next year: {predicted_price}')
    
    individual_tree_predictions = np.array([tree.predict(X_district) for tree in model.estimators_])
    predicted_price_std_dev = individual_tree_predictions.std()

    #Calculate prediction range 
    prediction_range = [predicted_price - predicted_price_std_dev, predicted_price + predicted_price_std_dev]
    st.write(f'Prediction range: {prediction_range}')

    
    district_df = pd.DataFrame({
        'district': df['district'].unique(),
        'latitude': df.groupby('district')['latitude'].mean().values,
        'longitude': df.groupby('district')['longitude'].mean().values
    })

    districts_for_prediction = df['district'].unique()

    district_predictions = pd.DataFrame({
        # 'district': districts_for_prediction,
        # 'predicted_price_sqm_next_year': model.predict(df[df['district'].isin(districts_for_prediction)][features])
        'district': df['district'],
        'latitude': df['latitude'],
        'longitude': df['longitude'],
        'predicted_price_sqm_next_year': model.predict(df[features])
    })

    print("============district_predictions shape========", district_predictions.shape)
    

    latitude_list = district_predictions.groupby('district', as_index=False)['latitude'].mean().values
    longitude_list = district_predictions.groupby('district', as_index=False)['longitude'].mean().values

    #print(latitude_list)
    #print(longitude_list)
    #predicted_df['latitude'] = df.groupby('district')['latitude'].mean().values
    #predicted_df['longitude'] = df.groupby('district')['longitude'].mean().values

    # Set predicted prices for corresponding districts
    # predicted_prices = predicted_df.loc[predicted_df['district'].isin(districts), 'predicted_price_sqm_next_year'] 


    fig = px.density_mapbox(district_predictions, lat='latitude', lon='longitude', z='predicted_price_sqm_next_year',
                            radius=10, center=dict(lat=13.736717, lon=100.523186), zoom=10,
                            mapbox_style="stamen-terrain", title='Predicted Prices Heatmap')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    st.header('Predicted Prices Heatmap')
    st.plotly_chart(fig)


if __name__ == '__main__':
    app()
