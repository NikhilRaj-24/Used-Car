import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Streamlit app
st.title('Used Car Price Prediction')

# User inputs for prediction
st.header("Enter Car Details")

# Load example data to extract options
df = pd.read_csv('final-streamlit(non-dup).csv', encoding='latin1')

df.drop_duplicates(inplace=True)

# Preprocess the data
X = df.drop(['Price_numeric'], axis=1)
y = df['Price_numeric']

# Identify numeric, one-hot, and binary features
num_features = X.select_dtypes(exclude="object").columns
onehot_columns = ['Fuel Type', 'Transmission', 'City']
binary_columns = ['Make', 'Model', 'Variant']

# Define preprocessing steps
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()
binary_transformer = BinaryEncoder()

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features),
        ("BinaryEncoder", binary_transformer, binary_columns)
    ]  
)

# Apply the transformations
X = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Select Make
make = st.selectbox('Make', sorted(df['Make'].unique()), index=sorted(df['Make'].unique()).index('Maruti Suzuki'))

# Filter models based on selected make
filtered_models = df[df['Make'] == make]['Model'].unique()

# Retrieve the previously selected model (if available)
if 'model_name' in st.session_state:
    previous_model = st.session_state['model_name']
else:
    previous_model = filtered_models[0]  # Default to the first model

# Handle case where previous model is not available for the selected make
if previous_model in filtered_models:
    model_name = st.selectbox('Model', sorted(filtered_models), index=sorted(filtered_models).index(previous_model))
else:
    st.warning(f"Model '{previous_model}' not available for Make '{make}'. Please select a different model.")
    model_name = st.selectbox('Model', sorted(filtered_models))

# Store the selected model in session state for future reference
st.session_state['model_name'] = model_name

# Filter variants based on selected model
filtered_variants = df[(df['Make'] == make) & (df['Model'] == model_name)]['Variant'].unique()

# Retrieve the previously selected variant (if available)
if 'variant' in st.session_state:
    previous_variant = st.session_state['variant']
else:
    previous_variant = filtered_variants[0]  # Default to the first variant

# Handle case where previous variant is not available for the selected model
if previous_variant in filtered_variants:
    variant = st.selectbox('Variant', sorted(filtered_variants), index=sorted(filtered_variants).index(previous_variant))
else:
    variant = st.selectbox('Variant', sorted(filtered_variants))

# Store the selected variant in session state for future reference
st.session_state['variant'] = variant

# Filter Transmission based on selected Make, Model, and Variant
filtered_transmissions = df[(df['Make'] == make) & (df['Model'] == model_name) & (df['Variant'] == variant)]['Transmission'].unique()

# Transmission selection with session state check
if 'transmission' in st.session_state:
    previous_transmission = st.session_state['transmission']
else:
    previous_transmission = filtered_transmissions[0]  # Default to the first transmission option

if previous_transmission in filtered_transmissions:
    transmission = st.selectbox('Transmission', sorted(filtered_transmissions), index=sorted(filtered_transmissions).index(previous_transmission))
else:
    transmission = st.selectbox('Transmission', sorted(filtered_transmissions))

st.session_state['transmission'] = transmission

# Filter Fuel Type based on selected Make, Model, and Variant
filtered_fuel_types = df[(df['Make'] == make) & (df['Model'] == model_name) & (df['Variant'] == variant)]['Fuel Type'].unique()

# Fuel Type selection with session state check
if 'fuel_type' in st.session_state:
    previous_fuel_type = st.session_state['fuel_type']
else:
    previous_fuel_type = filtered_fuel_types[0]  # Default to the first fuel type option

if previous_fuel_type in filtered_fuel_types:
    fuel_type = st.selectbox('Fuel Type', sorted(filtered_fuel_types), index=sorted(filtered_fuel_types).index(previous_fuel_type))
else:
    fuel_type = st.selectbox('Fuel Type', sorted(filtered_fuel_types))

st.session_state['fuel_type'] = fuel_type

# Select City
city = st.selectbox('City', sorted(df['City'].unique()))

# Numeric inputs: Age and Distance with default values
age = st.number_input('Enter Age (in years)', min_value=0, max_value=100, value=9)
distance = st.number_input('Enter Distance Driven (in km)', min_value=0, max_value=int(df['Distance_numeric'].max()), value=122847)


# Check if the selected options are valid before making predictions
if model_name in filtered_models and variant in filtered_variants:
    # Combine all inputs into a DataFrame
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model_name],
        'Variant': [variant],
        'Transmission': [transmission],
        'Fuel Type': [fuel_type],
        'City': [city],
        'Age': [age],
        'Distance_numeric': [distance]
    })

    # Check the frequency count for the selected car
    car_count = df[
        (df['Make'] == make) & 
        (df['Model'] == model_name) & 
        (df['Variant'] == variant) &
        (df['Fuel Type'] == fuel_type)
    ].shape[0]

    # Preprocess user input
    input_transformed = preprocessor.transform(input_data)

    # Predict price
    predicted_price = model.predict(input_transformed)[0]

    # Calculate the mean price for similar cars in the dataset
    mean_price = df[
        (df['Make'] == make) & 
        (df['Model'] == model_name) & 
        (df['Variant'] == variant) &
        (df['Fuel Type'] == fuel_type)
    ]['Price_numeric'].mean()

    if car_count < 5:
        st.write(f"Predicted Car Price: ₹{round(predicted_price)}")
        st.warning(f"The selected car model '{model_name} {variant}' has only {car_count} data points. We do not have enough data to predict accurately.")
    else:
        # Display predicted and mean prices directly
        st.write(f"Predicted Car Price: ₹{round(predicted_price)}")

        if pd.notnull(mean_price):
            st.write(f"Mean Car Price for this car: ₹{round(mean_price)}")
            st.write(f"Count of similar cars in the dataset: {car_count}")
        else:
            st.write("No similar cars found in the dataset to calculate the mean price.")
else:
    st.warning("Selected Model and/or Variant is not valid for the chosen Make. Please correct the selections to proceed.")
