# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    return pd.read_csv(file_path)

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(data):
    # Handle missing data using imputation
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Deal with outlier data (you can add outlier handling code here)
    
    return data_imputed

# Function to split data into training and testing sets (5 pts)
def split_data(data): 
    # Split data into training (80%) and testing (20%) sets
    X = data.drop("Outcome", axis=1)  
    y = data["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    # Train a or many models with hyperparameter tuning
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
    
    return model, hyperparameters

# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
    # Evaluate the model 
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Affichage des résultats
    print(f"Précision : {precision}")
    print(f"Rappel : {recall}")
    print(f"Score F1 : {f1}")
    print(f"Exactitude : {accuracy}")

    return precision, recall, f1, accuracy

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    # Deploy the best model using Streamlit
    st.title("Déploiement du modèle de prédiction du diabète")
    
    # Add input fields for features required by the model
    pregnancies = st.slider("Nombre de grossesses", 0, 17, 3)
    glucose = st.slider("Concentration de glucose", 0, 200, 100)
    blood_pressure = st.slider("Pression artérielle", 0, 122, 70)
    skin_thickness = st.slider("Épaisseur du pli cutané", 0, 99, 20)
    insulin = st.slider("Insuline", 0, 846, 100)
    bmi = st.slider("Indice de masse corporelle (BMI)", 0.0, 67.1, 30.0)
    diabetes_pedigree_function = st.slider("Fonction de pédigrée du diabète", 0.078, 2.42, 0.5)
    age = st.slider("Âge", 21, 81, 40)
    
    # Create a dataframe with the values entered by the user
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })
    
    # Make a prediction with the model
    prediction = model.predict(user_data)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.error("Le modèle prédit la présence de diabète.")
    else:
        st.success("Le modèle prédit l'absence de diabète.")

# Main function
def main():
    # Load data
    data = load_data("diabetes.csv")  # Replace with the actual path
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    
    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
    
    # Deploy the model (bonus)
    deploy_model(best_model, X_test)

    
if __name__ == "__main__":
    main()
