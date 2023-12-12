# AtmosAware Streamlit Project

AtmosAware is a Streamlit-based web application for performing regression analysis and predicting PM2.5 (particulate matter) levels based on various environmental factors.

## Features

- Perform regression analysis using different regression models.
- Predict PM2.5 values based on user-input environmental factors.
- Visualization of actual vs. predicted PM2.5 values.
- Evaluation metrics for regression models.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/AtmosAware.git
    ```

2. Navigate to the project directory:

    ```bash
    cd AtmosAware
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Access the web application by opening the provided URL in your browser.

3. Use the sidebar sliders to input environmental factors and predict PM2.5 levels.
   
4. Explore different regression models available in the app and observe their evaluation metrics and visualizations.

## File Structure

- `app.py`: Main Streamlit application file.
- `data_processing.py`: Module for data loading and preprocessing.
- `regression_models.py`: Module containing functions for various regression models.
- `README.md`: Documentation for the project.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
