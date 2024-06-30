# Decision Tree Classifier Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://decision-tree-builder.streamlit.app/)

This web application allows users to upload a CSV file and train a Decision Tree classifier with customizable hyperparameters. The application provides tools to visualize and evaluate the trained model, making it easier to understand its performance and structure.

## Features

- **File Upload:** Upload a CSV file containing your dataset.
- **Hyperparameter Tuning:** Customize various hyperparameters for the Decision Tree model, including criterion, splitter, maximum depth, minimum samples split, test size, and random state.
- **Data Preview:** View a preview of the uploaded data.
- **Model Training:** Train a Decision Tree model using the selected features and target column.
- **Model Evaluation:** Display the accuracy of the trained model.
- **Tree Visualization:** Visualize the trained Decision Tree as a static image and an interactive graph.
- **Python Code Generation:** Generate and display the Python code used to create the Decision Tree.

## Usage

1. **Upload a CSV File:** Use the file uploader to select and upload your dataset in CSV format.
2. **Select Target Column:** Choose the target column to predict from the dataset.
3. **Select Feature Columns:** Optionally, choose the feature columns to use for training the model.
4. **Configure Hyperparameters:** Adjust the hyperparameters for the Decision Tree model as needed.
5. **View Data:** Preview the uploaded dataset to ensure it is loaded correctly.
6. **Train Model:** Train the Decision Tree model with the configured settings.
7. **Evaluate Model:** Check the accuracy of the trained model.
8. **Visualize Tree:** View the Decision Tree structure as a static image or an interactive graph.
9. **Generate Code:** Display the Python code used for training the model.

## Hyperparameters

- **Criterion:** The function to measure the quality of a split (`gini`, `entropy`, `log_loss`).
- **Splitter:** The strategy used to choose the split at each node (`best`, `random`).
- **Max Depth:** The maximum depth of the tree.
- **Min Samples Split:** The minimum number of samples required to split an internal node.
- **Test Size:** The proportion of the dataset to include in the test split.
- **Random State:** The seed used by the random number generator.

## Tabs

- **Accuracy:** Displays the accuracy of the trained model.
- **Interactive Graph:** Shows an interactive visualization of the Decision Tree.
- **Static Image:** Provides a static image of the Decision Tree.
- **Python Code:** Displays the Python code used to create the Decision Tree.

## Side Panel

- **Hyperparameters Section:** Customize hyperparameters and select feature columns.
- **Random State Toggle:** Optionally set a random seed for reproducibility.
- **Maximum Depth Toggle:** Enable or disable the setting of a maximum depth for the tree.

## Buttons

- **Show Data:** Display a preview of the uploaded dataset.
- **Train Model:** Train the Decision Tree model.
- **Show Tree:** Generate and display the Decision Tree visualization.
