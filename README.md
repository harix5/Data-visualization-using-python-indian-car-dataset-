# Vehicle Data Visualization Dashboard

This repository contains a Streamlit-based data visualization dashboard for exploring vehicle specifications, pricing patterns, and performance characteristics. The project focuses on clean preprocessing, intuitive filtering, and clear visual insights, serving as a practical foundation for data analysis and early-stage machine learning workflows.

## Overview
The dashboard provides an interactive way to analyze automotive data, enabling users to explore trends across price, engine specifications, fuel types, and body categories. All visualizations are built using standard Python data libraries, making the project easy to understand, extend, and integrate into larger analytics pipelines.

## Key Features
- Interactive filters for Make, Body Type, Fuel Type, and Price Range  
- Automatic price standardization and conversion into lakhs  
- Clean light-themed interface for readability  
- Multiple comparative visualizations for price, power, torque, and displacement  
- Correlation heatmap for identifying relationships between numerical attributes  
- Export of filtered data directly from the dashboard  

## 🌐 Live Demo
Access the hosted application:

🔗 **Live Dashboard:** [LIVE DEMO ](https://shorturl.at/Fhpq4)

## Visualizations
- Distribution of Body Types
  <img width="984" height="584" alt="output - VECHICLE COUNT BBY BODY TYPE" src="https://github.com/user-attachments/assets/f5bb2711-3e3d-41db-afab-20fef86e2ff2" />

- Distribution of Fuel Types
  <img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/38b824fc-b70a-4af0-b3f6-a68a9a6c23be" />
 
- Price Distribution (in Lakhs)
  <img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/ab4710ba-2165-4043-b1a3-e1eac8a21901" />

- Price Variation by Body Type
  <img width="1380" height="484" alt="image" src="https://github.com/user-attachments/assets/bee11fff-5584-4bc5-be67-d0018a6d7415" />

- Power vs. Torque Segmented by Body type
  <img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/e1f442c8-5181-4d04-b436-bca53e6157d3" />


- Power Output by Fuel Type
  
  <img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/dc935f8e-6f1b-4dd4-b5cb-401d81bb2e0b" />

- Displacement vs. Price (with trend line)
  <img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/f79b0e8d-206f-4663-a9a4-39e760fdcbd4" />

- Core Specifications - Pair Wise Comparison
  <img width="983" height="1022" alt="image" src="https://github.com/user-attachments/assets/33f1e385-e346-4585-ba6c-d18c7c39d1a4" />

- Correlation Heatmap
  <img width="1104" height="784" alt="image" src="https://github.com/user-attachments/assets/78b36a92-e402-4314-b3a0-e300eea5bc3d" />


These visualizations support exploratory data analysis (EDA), a core step before applying machine learning models.

## Project Structure
dashboard.py # Main Streamlit application
requirements.txt # Python dependencies
car_dataset.csv #dataset
README.md # Project documentation
LICENSE # MIT License

## Purpose
This project is designed to:
- Demonstrate practical data cleaning and preprocessing  
- Provide a structured approach to EDA  
- Help beginners transition into machine learning by understanding data behavior  
- Serve as a template for future dashboards or analytics tools  

## Technologies Used
- Python  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

## Dataset
The dashboard supports any dataset with similar columns (Make, Price, Power, Fuel_Type, Body_Type, etc.). The included `car_dataset.csv` serves as an example input file.

## License
This project is licensed under the MIT License. Refer to the `LICENSE` file for details.

## Contributing
Contributions, improvements, and feature additions are welcome. Fork the repository and submit a pull request.

## Acknowledgment
If you find this project useful, consider starring the repository to support further development.


