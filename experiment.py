
import sys
import logging
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_synthetic_dataset():
    """
    Create a synthetic dataset for demonstration purposes.
    In a real scenario, this would be replaced with actual data loading.
    """
    logger.info("Creating synthetic dataset...")
    try:
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        logger.info(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    except Exception as e:
        logger.error(f"Error creating synthetic dataset: {str(e)}")
        sys.exit(1)

def preprocess_data(X, y):
    """
    Preprocess the data by splitting and scaling.
    """
    logger.info("Preprocessing data...")
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Data scaling completed")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        sys.exit(1)

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    logger.info("Training Random Forest model...")
    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    """
    logger.info("Evaluating model performance...")
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Generate detailed classification report
        report = classification_report(y_test, y_pred, output_dict=False)
        logger.info("Classification Report:\n" + report)
        
        return accuracy
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        sys.exit(1)

def analyze_feature_importance(model, feature_names=None):
    """
    Analyze and log feature importance from the trained model.
    """
    logger.info("Analyzing feature importance...")
    try:
        importances = model.feature_importances_
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Sort features by importance
        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Log top 10 most important features
        logger.info("Top 10 most important features:")
        for feature, importance in feature_importance[:10]:
            logger.info(f"  {feature}: {importance:.4f}")
            
        return feature_importance
    except Exception as e:
        logger.error(f"Error during feature importance analysis: {str(e)}")
        sys.exit(1)

def main():
    """
    Main function to execute the complete machine learning pipeline.
    """
    logger.info("Starting machine learning experiment...")
    
    try:
        # Step 1: Create or load dataset
        X, y = create_synthetic_dataset()
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        # Step 3: Train model
        model = train_model(X_train, y_train)
        
        # Step 4: Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        
        # Step 5: Analyze feature importance
        feature_importance = analyze_feature_importance(model)
        
        # Final summary
        logger.info("Experiment completed successfully!")
        logger.info(f"Final model accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'model_trained': True
        }
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Execute the main function
    result = main()
    
    # Exit with success code
    sys.exit(0)
