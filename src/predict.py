import pandas as pd


def predict_and_save(model, test, test_ids, output_path):
    """
    Generate predictions on test dataset and save output file.

    Parameters:
        model: Trained pipeline
        test (DataFrame): Test features
        test_ids (Series): ID column
        output_path (str): File path to save predictions
    """

    # Predict probability of default
    probs = model.predict_proba(test)[:, 1]

    # Create output DataFrame
    output = pd.DataFrame({
        "ID": test_ids,
        "Probability_of_Default": probs
    })

    # Save to CSV
    output.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")