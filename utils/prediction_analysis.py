import numpy as np
import pandas as pd

def calculate_waste_and_lost_sales_for_models(data_dict: dict):
    # Extract the actuals dataframe from the dictionary
    y_true = data_dict.get('Actual')
    if y_true is None:
        raise ValueError("The dictionary must contain a key 'actuals' with the actual sales data.")
    
    # Convert the index to datetime if it's not already
    y_true.index = pd.to_datetime(y_true.index)
    
    # Initialize a dictionary to hold results for each model
    results_dict = {}
    
    # Iterate over each model in the dictionary
    for model_name, y_pred in data_dict.items():
        if model_name == 'Actual':
            continue  # Skip the actuals key
        
        # Convert index to datetime if it's not already
        y_pred.index = pd.to_datetime(y_pred.index)
        
        # Aggregate by day
        daily_true = y_true.resample('D').sum()
        daily_pred = y_pred.resample('D').sum()
        
        # Calculate waste and lost sales
        waste = np.maximum(daily_pred - daily_true, 0)
        lost_sales = np.maximum(daily_true - daily_pred, 0)
        
        # Calculate total production and total demand
        total_production = daily_pred.sum(axis=1)
        total_demand = daily_true.sum(axis=1)
        
        # Calculate waste and lost sales percentages
        waste_percentage = (waste.sum(axis=1) / total_production) * 100
        lost_sales_percentage = (lost_sales.sum(axis=1) / total_demand) * 100
        
        # Combine results in a DataFrame
        results = pd.DataFrame({
            "Total Production": total_production,
            "Total Demand": total_demand,
            "Waste Volume": waste.sum(axis=1),
            "Lost Sales Volume": lost_sales.sum(axis=1),
            "Waste Percentage": waste_percentage,
            "Lost Sales Percentage": lost_sales_percentage
        })
        
        # Store the results in the results dictionary under the model name
        results_dict[model_name] = results
        #print(model_name, results)
    
    return results_dict

def generate_summary(daily_results_dict: dict):
    summary_dict = {}
    
    # Iterate over each model's daily results
    for model_name, results in daily_results_dict.items():
        # Calculate overall summary for the entire period
        total_production_summary = results["Total Production"].sum()
        total_demand_summary = results["Total Demand"].sum()
        total_waste_volume_summary = results["Waste Volume"].sum()
        total_lost_sales_volume_summary = results["Lost Sales Volume"].sum()
        
        # Waste and Lost Sales percentages over the entire period
        waste_percentage_summary = (total_waste_volume_summary / total_production_summary) * 100
        lost_sales_percentage_summary = (total_lost_sales_volume_summary / total_demand_summary) * 100
        
        # Create a summary row and append it to the DataFrame
        summary = pd.DataFrame({
            "Total Production": [total_production_summary],
            "Total Demand": [total_demand_summary],
            "Waste Volume": [total_waste_volume_summary],
            "Lost Sales Volume": [total_lost_sales_volume_summary],
            "Waste Percentage": [waste_percentage_summary],
            "Lost Sales Percentage": [lost_sales_percentage_summary]
        }, index=["Summary"])
        
        # Append the summary row to the results DataFrame
        results_with_summary = pd.concat([results, summary])
        
        # Store the summary results in the summary dictionary under the model name
        summary_dict[model_name] = summary
        print(model_name, summary)
    
    return summary_dict


if __name__ == "__main__":
    # Example usage
    data_dict = {
        'Actual': pd.DataFrame({
            "Item1": [100, 150, 200, 250, 300, 350, 400],
            "Item2": [120, 130, 140, 160, 180, 190, 200],
            "Item3": [180, 190, 200, 210, 220, 230, 240],
            "Item4": [220, 230, 240, 250, 260, 270, 280]
        }, index=pd.date_range("2024-08-01", periods=7, freq='H')),

        'model_1': pd.DataFrame({
            "Item1": [110, 140, 195, 255, 310, 340, 390],
            "Item2": [115, 135, 145, 150, 185, 185, 195],
            "Item3": [185, 185, 210, 200, 225, 225, 235],
            "Item4": [225, 220, 235, 260, 275, 265, 275]
        }, index=pd.date_range("2024-08-01", periods=7, freq='H')),

        'model_2': pd.DataFrame({
            "Item1": [100, 160, 180, 240, 320, 360, 410],
            "Item2": [125, 135, 150, 170, 190, 200, 210],
            "Item3": [190, 200, 210, 220, 230, 240, 250],
            "Item4": [230, 240, 250, 260, 270, 280, 290]
        }, index=pd.date_range("2024-08-01", periods=7, freq='H'))
    }

    results_dict = calculate_waste_and_lost_sales_for_models(data_dict)

    # To view the results for each model:
    for model_name, result_df in results_dict.items():
        print(f"Results for {model_name}:")
        print(result_df, "\n")
