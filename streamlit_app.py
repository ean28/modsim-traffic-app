import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, norm
import plotly.express as px


st.set_page_config(page_title="Cafe Traffic Prediction", layout="wide")

# Function to generate synthetic data
def generate_data(n_samples):
    np.random.seed(42)
    time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening'], size=n_samples, p=[0.35, 0.32, 0.33])
    day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                    size=n_samples, p=[0.15, 0.1, 0.1, 0.1, 0.225, 0.225, 0.1])
    weather = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=n_samples, p=[0.5, 0.2, 0.3])
    promotion = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.2, 0.8])
    special_event = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.1, 0.9])

    # correlations
    base_traffic = np.array([50 if t == 'Morning' else 80 if t == 'Afternoon' else 40 for t in time_of_day])
    weather_impact = np.array([10 if w == 'Sunny' else -20 if w == 'Rainy' else 0 for w in weather])
    promotion_impact = np.array([20 if p == 'Yes' else 0 for p in promotion])
    event_impact = np.array([30 if e == 'Yes' else 0 for e in special_event])
    random_noise = np.random.normal(0, 6, size=n_samples)

    customer_traffic = base_traffic + weather_impact + promotion_impact + event_impact + random_noise
    customer_traffic = np.clip(customer_traffic, 0, None).astype(int)

    return pd.DataFrame({
        'Time of Day': time_of_day,
        'Day of Week': day_of_week,
        'Weather': weather,
        'Promotion': promotion,
        'Special Event': special_event,
        'Customer Traffic': customer_traffic
    })

# Sidebar controls
st.sidebar.header("Controls")
n_samples = st.sidebar.slider("Number of Samples", min_value=500, max_value=5000, step=100, value=1000)
test_size = st.sidebar.slider("Test Size (as %)", min_value=10, max_value=50, step=5, value=20) / 100
regenerate = st.sidebar.button("Regenerate Data")
st.sidebar.divider()
st.sidebar.header("Simulation Controls")
scenario = st.sidebar.selectbox("Select Scenario", 
                                ["Morning Rush", "Afternoon Dip", "Rainy Day", "Promotion Day", "Special Event Day"])

#---------------------------------------------------
if "data" not in st.session_state or regenerate:
    st.session_state["data"] = generate_data(n_samples)

data = st.session_state["data"]

if regenerate:
    st.session_state["data"] = generate_data(n_samples)
    data = st.session_state["data"]

# Ensure data exists in session state before displaying the split info
if "data" in st.session_state:
    data = st.session_state["data"]
    
    st.title("Modeling and Simulation of Customer Traffic in a Cafe")
    st.markdown(f"""This project aims to predict the peak hours for sales at a supposed cafe based on various factors. 
                The goal of this project is to focus on modeling and simulation using Python by generating synthetic data and applying machine learning models. 
                The goal is to gain hands-on experience with various libraries.""", unsafe_allow_html=True)
    st.markdown(f'<br><font style="font-size: 12px;"> by De las Llagas, Llorente, and Aguilar </strong>', unsafe_allow_html=True)
    st.divider()

    # Dataset Split Information only displayed after the data is generated or regenerated
    st.subheader("Dataset Split Information")

    train_size = 1 - test_size
    n_train_samples = int(n_samples * train_size)
    n_test_samples = int(n_samples * test_size)

    train_percentage = (n_train_samples / n_samples) * 100
    test_percentage = (n_test_samples / n_samples) * 100

    # Columns for displaying the dataset split information
    view_total, view_testing, view_training = st.columns(3)

    # Display Total Samples
    with view_total:
        st.markdown(f'Total Samples: <br><strong style="font-size: 25px;"> {n_samples} </strong>', unsafe_allow_html=True)

    # Display Testing Samples
    with view_testing:
        st.markdown(f'Test Samples: <br><strong style="font-size: 25px;">{n_test_samples} ({test_percentage}%)</strong>', unsafe_allow_html=True)

    # Display Training Samples
    with view_training:
        st.markdown(f'Training Samples: <br><strong style="font-size: 25px;">{n_train_samples} ({train_percentage}%)</strong>', unsafe_allow_html=True)

    # Display dataset
    st.subheader("Generated Dataset")
    st.dataframe(data)

# Function to preprocess data
def preprocess_data(data):
    X = data[['Time of Day', 'Day of Week', 'Weather', 'Promotion', 'Special Event']]
    y = data['Customer Traffic']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), ['Time of Day', 'Day of Week', 'Weather', 'Promotion', 'Special Event'])])
    return X, y, preprocessor

# Function to train and evaluate models
def train_models(X_train, y_train, X_test, y_test, preprocessor):
    models = {
        'Linear Regression': Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        'Random Forest': Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'model': model
        }
    return results
st.divider()
# Interactive visualization selection
st.subheader("Data Visualizations")

# Create radio buttons for 2D or 3D selection
plot_mode = st.radio("Choose Plot Mode", ['2D', '3D'], index=1)

# Create two columns to display the dropdowns side by side
col1, col2 = st.columns(2)

# Dropdown for Plot 1 type (Column 1)
with col1:
    plot_1_type = st.selectbox(
        "Choose Plot 1 Type",
        [
            'Barplot: Time of Day vs Customer Traffic',
            'Boxplot: Weather vs Customer Traffic',
            'Lineplot: Promotion vs Customer Traffic'
        ]
    )

# Dropdown for Plot 2 type (Column 2)
with col2:
    plot_2_type = st.selectbox(
        "Choose Plot 2 Type",
        [
            'Barplot: Day of Week vs Customer Traffic',
            'Boxplot: Time of Day vs Customer Traffic',
            'Lineplot: Day of Week vs Customer Traffic'
        ]
    )

# Set the layout for two plots side by side
col1, col2 = st.columns(2)

# Plot 1
with col1:
    if plot_mode == '2D':
        if plot_1_type == 'Barplot: Time of Day vs Customer Traffic':
            # 2D Barplot for Time of Day vs Customer Traffic
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=data, x='Time of Day', y='Customer Traffic', ci=None, palette='viridis', ax=ax1)
            ax1.set_title("Traffic by Time of Day")
            ax1.set_ylabel("Average Traffic")
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
        elif plot_1_type == 'Boxplot: Weather vs Customer Traffic':
            # 2D Boxplot for Weather vs Customer Traffic
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='Weather', y='Customer Traffic', palette='coolwarm', ax=ax1)
            ax1.set_title("Traffic Distribution by Weather")
            ax1.set_ylabel("Customer Traffic")
            st.pyplot(fig1)
        elif plot_1_type == 'Lineplot: Promotion vs Customer Traffic':
            # 2D Lineplot for Promotion vs Customer Traffic
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=data, x='Promotion', y='Customer Traffic', ax=ax1)
            ax1.set_title("Traffic by Promotion")
            ax1.set_ylabel("Customer Traffic")
            st.pyplot(fig1)
    else:
        if plot_1_type == 'Barplot: Time of Day vs Customer Traffic':
            # 3D Scatter Plot for Time of Day vs Customer Traffic
            fig1 = px.scatter_3d(
                data,
                x='Time of Day',
                y='Weather',
                z='Customer Traffic',
                color='Promotion',
                title="3D View of Customer Traffic by Time of Day and Weather"
            )
            fig1.update_layout(
                scene=dict(
                    xaxis_title='Time of Day',
                    yaxis_title='Weather Condition',
                    zaxis_title='Customer Traffic Volume'
                )
            )
            st.plotly_chart(fig1)
        elif plot_1_type == 'Boxplot: Weather vs Customer Traffic':
            # 3D Scatter Plot for Weather vs Customer Traffic
            fig1 = px.scatter_3d(
                data,
                x='Weather',
                y='Time of Day',
                z='Customer Traffic',
                color='Promotion',
                title="3D View of Customer Traffic by Weather and Time of Day"
            )
            st.plotly_chart(fig1)
        elif plot_1_type == 'Lineplot: Promotion vs Customer Traffic':
            # 3D Scatter Plot for Promotion vs Customer Traffic
            fig1 = px.scatter_3d(
                data,
                x='Promotion',
                y='Weather',
                z='Customer Traffic',
                color='Time of Day',
                title="3D View of Customer Traffic by Promotion and Weather"
            )
            st.plotly_chart(fig1)

# Plot 2
with col2:
    if plot_mode == '2D':
        if plot_2_type == 'Barplot: Day of Week vs Customer Traffic':
            # 2D Barplot for Day of Week vs Customer Traffic
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=data, x='Day of Week', y='Customer Traffic', ci=None, palette='viridis', ax=ax2)
            ax2.set_title("Traffic by Day of Week")
            ax2.set_ylabel("Average Traffic")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)
        elif plot_2_type == 'Boxplot: Time of Day vs Customer Traffic':
            # 2D Boxplot for Time of Day vs Customer Traffic
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='Time of Day', y='Customer Traffic', palette='coolwarm', ax=ax2)
            ax2.set_title("Traffic Distribution by Time of Day")
            ax2.set_ylabel("Customer Traffic")
            st.pyplot(fig2)
        elif plot_2_type == 'Lineplot: Day of Week vs Customer Traffic':
            # 2D Lineplot for Day of Week vs Customer Traffic
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=data, x='Day of Week', y='Customer Traffic', ax=ax2)
            ax2.set_title("Traffic by Day of Week")
            ax2.set_ylabel("Customer Traffic")
            st.pyplot(fig2)
    else:
        if plot_2_type == 'Barplot: Day of Week vs Customer Traffic':
            # 3D Scatter Plot for Day of Week vs Customer Traffic
            fig2 = px.scatter_3d(
                data,
                x='Day of Week',
                y='Weather',
                z='Customer Traffic',
                color='Promotion',
                title="3D View of Customer Traffic by Day of Week and Weather"
            )
            st.plotly_chart(fig2)
        elif plot_2_type == 'Boxplot: Time of Day vs Customer Traffic':
            # 3D Scatter Plot for Time of Day vs Customer Traffic
            fig2 = px.scatter_3d(
                data,
                x='Time of Day',
                y='Weather',
                z='Customer Traffic',
                color='Promotion',
                title="3D View of Customer Traffic by Time of Day and Weather"
            )
            st.plotly_chart(fig2)
        elif plot_2_type == 'Lineplot: Day of Week vs Customer Traffic':
            # 3D Scatter Plot for Day of Week vs Customer Traffic
            fig2 = px.scatter_3d(
                data,
                x='Day of Week',
                y='Weather',
                z='Customer Traffic',
                color='Promotion',
                title="3D View of Customer Traffic by Day of Week and Weather"
            )
            st.plotly_chart(fig2)

# Train-test split
X, y, preprocessor = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.divider()
# Train models
st.subheader("Model Comparison")
model_results = train_models(X_train, y_train, X_test, y_test, preprocessor)

# Display model performances
comparison = pd.DataFrame({name: metrics for name, metrics in model_results.items()}).T.drop('model', axis=1)
st.table(comparison)

# Select best model
best_model_name = comparison['R²'].idxmax()
best_model = model_results[best_model_name]['model']
st.write(f"**Best Model:** {best_model_name} with R² = {comparison.loc[best_model_name, 'R²']:.2f}")
st.divider()

# Simulation
st.subheader("Interactable Simulation of Customer Traffic")
sim_col = st.columns(2)
with sim_col[0]:
    time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening'])
    day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weather = st.selectbox("Weather", ['Sunny', 'Rainy', 'Cloudy'])
with sim_col[1]:
    promotion = st.selectbox("Promotion Active", ['Yes', 'No'])
    special_event = st.selectbox("Special Event", ['Yes', 'No'])

simulation_data = pd.DataFrame({
    'Time of Day': [time_of_day],
    'Day of Week': [day_of_week],
    'Weather': [weather],
    'Promotion': [promotion],
    'Special Event': [special_event]
})

simulation_transformed = best_model.named_steps['preprocessor'].transform(simulation_data)
predicted_traffic = best_model.named_steps['regressor'].predict(simulation_transformed)

# Create centered columns
fact_center = st.columns([1, 2, 1])  # Middle column will be the center one

with fact_center[1]:  # Content will be placed in the center column
    st.write("### Factors:")
    st.write(simulation_data)
    st.write(f"**Predicted Customer Traffic:** {predicted_traffic[0]:.0f}")

# Add a divider for better visual structure
st.divider()

# Function to simulate using Poisson distribution
def poisson_simulation(lambda_value, n_samples):
    return poisson.rvs(mu=lambda_value, size=n_samples)

# Function to simulate using Normal distribution
def normal_simulation(mean_value, std_dev_value, n_samples):
    return np.random.normal(loc=mean_value, scale=std_dev_value, size=n_samples)

# Scenario Parameters
if scenario == "Morning Rush":
    # High traffic in the morning
    lambda_value = 50  # High arrival rate for Poisson
    mean_value = 80  # High mean for Normal
    std_dev_value = 5  # Low variance for Normal
elif scenario == "Afternoon Dip":
    # Low traffic after lunch
    lambda_value = 25  # Moderate arrival rate for Poisson
    mean_value = 30  # Low mean for Normal
    std_dev_value = 10  # Slightly higher variance for Normal
elif scenario == "Rainy Day":
    # Reduced traffic on rainy days
    lambda_value = 10  # Low arrival rate for Poisson
    mean_value = 15  # Low mean for Normal
    std_dev_value = 15  # High variance for more randomness
elif scenario == "Promotion Day":
    # High traffic due to promotions
    lambda_value = 100  # Very high arrival rate for Poisson
    mean_value = 90  # High mean for Normal
    std_dev_value = 10  # Low variance for Normal
elif scenario == "Special Event Day":
    # Variable traffic due to special event
    lambda_value = 70  # High arrival rate for Poisson
    mean_value = 100  # Very high mean for Normal
    std_dev_value = 30  # High variance for high unpredictability

# Generate simulated traffic based on the selected scenario
simulated_traffic_poisson = poisson_simulation(lambda_value, n_samples)
simulated_traffic_normal = normal_simulation(mean_value, std_dev_value, n_samples)

b1,b2,b3 = st.columns([1,2,1])
with b2:
    # Compute permutation importance
    results = permutation_importance(best_model, X, y, n_repeats=10, random_state=42)
    st.header("Exploratory Data Analysis")
    # Plot importance
    st.subheader("Parameter Importance")
    plt.bar(X.columns, results.importances_mean)
    plt.ylabel('Importance')
    plt.title('Feature Importance (Permutation)')
    st.pyplot(plt)
    st.divider()

    # Display scenario information

    st.header("Evaluation and Analysis")
    st.subheader(f"Simulated Customer Traffic for '{scenario}' Scenario")

    # Create DataFrame for first 5 simulated values
    traffic_data = pd.DataFrame({
        'Poisson Traffic': simulated_traffic_poisson[:5],
        'Normal Traffic': simulated_traffic_normal[:5]
    }).T

    # Display the table of the first 5 simulated traffic values
    st.write("##### First 5 Simulated Traffic Values")
    st.dataframe(traffic_data)
    
fig_center = st.columns([1,2,1])
with fig_center[1]:
    # Visualization of simulated data
    fig, ax = plt.subplots(figsize=(11, 8))

    # Poisson distribution plot
    sns.histplot(simulated_traffic_poisson, kde=False, color='blue', ax=ax, label="Poisson")
    sns.histplot(simulated_traffic_normal, kde=True, color='green', ax=ax, label="Normal")
    
    # Set titles and labels
    ax.set_title(f"Customer Traffic Simulation for '{scenario}'")
    ax.set_xlabel("Traffic Volume")
    ax.set_ylabel("Frequency")
    ax.legend()

    st.pyplot(fig)

    p_col1, n_col2 = st.columns(2)

    with p_col1:
        st.write(f"**Simulation Statistics (Poisson):**")
        st.write(f"Mean: {np.mean(simulated_traffic_poisson):.2f}")
        st.write(f"Standard Deviation: {np.std(simulated_traffic_poisson):.2f}")
        st.write(f"Min: {np.min(simulated_traffic_poisson)}")
        st.write(f"Max: {np.max(simulated_traffic_poisson)}")
    with n_col2:
        st.write(f"**Simulation Statistics (Normal):**")
        st.write(f"Mean: {np.mean(simulated_traffic_normal):.2f}")
        st.write(f"Standard Deviation: {np.std(simulated_traffic_normal):.2f}")
        st.write(f"Min: {np.min(simulated_traffic_normal)}")
        st.write(f"Max: {np.max(simulated_traffic_normal)}")

    # Generate simulated traffic
    simulated_traffic_poisson = poisson_simulation(lambda_value, len(y_test))
    simulated_traffic_normal = normal_simulation(mean_value, std_dev_value, len(y_test))

    # Model predictions
    predicted_data = best_model.predict(X_test)[:len(data)]  # Ensure lengths match with original data

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, predicted_data)  # Use y_test instead of data for actual comparison
    mse = mean_squared_error(y_test, predicted_data)
    r2 = r2_score(y_test, predicted_data)

    st.divider()

    # Distribution Comparison: Original vs Simulated vs Predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y_test, kde=True, color='blue', label="Original Data", ax=ax)  # Use y_test for actual data
    sns.histplot(simulated_traffic_poisson, kde=True, color='green', label="Simulated (Poisson)", ax=ax)
    sns.histplot(predicted_data, kde=True, color='orange', label="Model Predictions", ax=ax)
    ax.set_title("Comparison of Distributions: Original vs Simulated vs Predicted")
    ax.set_xlabel("Customer Traffic Volume")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)
    st.write("This plot compares the distribution of the actual customer traffic (blue), the simulated traffic based on Poisson distribution (green), and the predicted traffic from the model (orange). The comparison helps assess how well the model predictions align with the actual data and the simulated scenarios.")

    # Scatter Plot: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predicted_data, ax=ax, color='purple', alpha=0.7)  # Use y_test for actual values
    ax.set_title("Actual vs Predicted Customer Traffic")
    ax.set_xlabel("Actual Customer Traffic")
    ax.set_ylabel("Predicted Customer Traffic")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Perfect prediction line
    st.pyplot(fig)
    st.write("This scatter plot shows the actual customer traffic versus the predicted traffic. The red dashed line represents a perfect prediction (where actual = predicted). Points along this line indicate accurate predictions, while those farther away represent larger errors in prediction.")

    # Residual Analysis
    residuals = y_test - predicted_data  
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', ax=ax)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    st.write("This plot shows the distribution of residuals (the difference between actual and predicted customer traffic). A well-performing model should have residuals that are normally distributed around zero, indicating unbiased predictions.")

    # Robustness Analysis: Noisy Data
    noisy_data = y_test + np.random.normal(0, 0.25 * np.std(y_test), size=len(y_test))  # Increase noise level

    # Evaluate model performance on noisy data
    predicted_noisy = best_model.predict(X_test)  # Predict using the model on X_test
    mae_noisy = mean_absolute_error(noisy_data, predicted_noisy)  # Compare with noisy data
    mse_noisy = mean_squared_error(noisy_data, predicted_noisy)
    r2_noisy = r2_score(noisy_data, predicted_noisy)

blank_col, noisy_col, normal_col = st.columns([0.45,1,1])
with noisy_col:
    # Display noisy data performance metrics
    st.write("### Model Performance on Noisy Data")
    st.write(f"**Mean Absolute Error (MAE) on Noisy Data:** {mae_noisy:.2f}")
    st.write(f"**Mean Squared Error (MSE) on Noisy Data:** {mse_noisy:.2f}")
    st.write(f"**R² Score on Noisy Data:** {r2_noisy:.2f}")
with normal_col:
    st.write("### Model Evaluation Metrics")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")


# Function to save model to a file
def save_model(model, model_name='best_model'):
    model_filename = f'{model_name}.pkl'
    with open(model_filename, 'wb') as f:
        joblib.dump(model, f)
    return model_filename

def save_model(model, model_name='model.pkl'):
    joblib.dump(model, model_name)
    return model_name

st.sidebar.divider()
st.sidebar.subheader("Download Options")
download_choice = st.sidebar.radio("Choose what to download", ("Dataset", "Trained Model"))

download_button = st.sidebar.button("Download")

if download_button:
    if download_choice == "Dataset":
        # Convert dataset to CSV and allow download
        csv = data.to_csv(index=False)
        st.sidebar.download_button(label="Download Dataset as CSV", data=csv, file_name="cafe_customer_traffic.csv", mime="text/csv")
    elif download_choice == "Trained Model":
        model_filename = save_model(best_model, model_name='cafe_traffic_model.pkl')
        st.sidebar.download_button(label="Download Trained Model", data=open(model_filename, 'rb').read(), file_name=model_filename, mime="application/octet-stream")