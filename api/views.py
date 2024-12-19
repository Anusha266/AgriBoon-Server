from django.shortcuts import render

# Create your views here.
import pandas as pd
import torch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
from django.conf import settings
from PIL import Image
from io import BytesIO
from torchvision import transforms
from datetime import datetime

# Load models at startup to avoid reloading every request


quality_mapping = {'low': 0.2, 'medium': 0.50, 'high': 1}

class PricePredictionAPIView(APIView):
    '''
    This API view is used to predict the price of a product based on the image of the product and the inflation rate.
    product_name: always takes in lower case with no spaces. ex: groundnut, rice, wheat, etc.
    inflation_rate: takes in the inflation rate of the country. ex(percentage values as decimal): 0.08,0.10
    '''
    
    def post(self, request, *args, **kwargs):
        # Step 1: Retrieve the image from the request
        image = request.FILES.get('image')
        inflation_rate = request.data.get('inflation_rate')
        product_name = request.data.get('product_name') 

        if not image or inflation_rate is None or product_name is None:
            return Response({"error": "Missing required fields."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            groundnut_model = torch.load('server/models/groundnut_model_epoch_300_with_softmax.pth')    
            min_price_model = joblib.load('server/models/dynamic_min_price_model.pkl')
            modal_price_model = joblib.load('server/models/dynamic_modal_price_model.pkl')
            max_price_model = joblib.load('server/models/dynamic_max_price_model.pkl')
        except Exception as e:
            return Response({"error": "Error loading models."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Step 2: Predict the quality from the image using the groundnut model
        
        
        quality = self.predict_quality(image,groundnut_model)

        
        if quality is None:
            return Response({"error": "Unable to predict quality from the image."}, status=status.HTTP_400_BAD_REQUEST)

        # Step 3: Get today's date
        today = pd.to_datetime('today').strftime('%Y-%m-%d')

    # Fetch the price_x data for today
        price_x_min = self.get_price_data(today, 'min_price', product_name)
        price_x_modal = self.get_price_data(today, 'modal_price', product_name)
        price_x_max = self.get_price_data(today, 'max_price', product_name)

        # Prepare the new data for prediction (using min_price, modal_price, and max_price)
        new_data_min = {
            'date': [today],
            'price_x': [price_x_min],
            'quality': [quality_mapping.get(quality, 0.5)],  # Map to numeric values
            'inflation_rate': [inflation_rate]
        }

        new_data_modal = {
            'date': [today],
            'price_x': [price_x_modal],
            'quality': [quality_mapping.get(quality, 0.5)],  # Map to numeric values
            'inflation_rate': [inflation_rate]
        }

        new_data_max = {
            'date': [today],
            'price_x': [price_x_max],
            'quality': [quality_mapping.get(quality, 0.5)],  # Map to numeric values
            'inflation_rate': [inflation_rate]
        }

        # Convert the dictionary into DataFrame
        new_data_df_min = pd.DataFrame(new_data_min)
        new_data_df_modal = pd.DataFrame(new_data_modal)
        new_data_df_max = pd.DataFrame(new_data_max)

        # Convert 'date' column to datetime and extract 'day_of_year'
        new_data_df_min['date'] = pd.to_datetime(new_data_df_min['date'], errors='coerce')
        new_data_df_min['day_of_year'] = new_data_df_min['date'].dt.dayofyear

        new_data_df_modal['date'] = pd.to_datetime(new_data_df_modal['date'], errors='coerce')
        new_data_df_modal['day_of_year'] = new_data_df_modal['date'].dt.dayofyear

        new_data_df_max['date'] = pd.to_datetime(new_data_df_max['date'], errors='coerce')
        new_data_df_max['day_of_year'] = new_data_df_max['date'].dt.dayofyear

        # Prepare features for prediction
        X_new_min = new_data_df_min[['day_of_year', 'quality', 'price_x', 'inflation_rate']]
        X_new_modal = new_data_df_modal[['day_of_year', 'quality', 'price_x', 'inflation_rate']]
        X_new_max = new_data_df_max[['day_of_year', 'quality', 'price_x', 'inflation_rate']]

        # Step 6: Make predictions using the models
        min_price_pred = min_price_model.predict(X_new_min)
        modal_price_pred = modal_price_model.predict(X_new_modal)
        max_price_pred = max_price_model.predict(X_new_max)

        # Return the predicted prices
        return min_price_pred[0], modal_price_pred[0], max_price_pred[0]
    
    def predict_quality(self, image,groundnut_model):
        """Predict the quality of the image using the groundnut model."""
        try:
            # Load the image
            img = Image.open(image)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            # Predict using the model
            groundnut_model.eval()
            with torch.no_grad():
                outputs = groundnut_model(img_tensor)
                _, predicted_class = torch.max(outputs, 1)
                # Return the predicted class (e.g., grade1, grade2, grade3)
                return ['low', 'medium', 'high'][predicted_class.item()]
        except Exception as e:
            print(f"Error in predicting quality: {e}")
            return None


    def get_price_data(self, date, price_type, product_name):
        """Fetch historical pricing data from CSV based on today's date.
        If the exact date is not found, return the closest available date before or after.
        price_type should be 'min_price', 'modal_price', or 'max_price'."""
        try:
            # Parse the provided date
            year, month, day = map(int, date.split('-'))
            target_date = pd.to_datetime(date)

        # Load the pricing data for the previous year
            last_year_data = pd.read_csv(f'server/data/{product_name}_price_data_{year-1}.csv')
            last_year_data['date'] = pd.to_datetime(last_year_data['date'], format='%Y-%m-%d')

            # Sort by date to make sure we can find the closest match
            last_year_data = last_year_data.sort_values(by='date')

            # Try to find the exact match first
            price_data = last_year_data[last_year_data['date'] == target_date]

            if not price_data.empty:
                # Return the specified price type (min_price, modal_price, or max_price)
                return price_data.iloc[0][price_type]  # Dynamically return the correct price column

            # If exact date is not found, find the closest available date
            before_data = last_year_data[last_year_data['date'] < target_date].iloc[-1:]  # Closest date before
            after_data = last_year_data[last_year_data['date'] > target_date].iloc[:1]  # Closest date after

            if not before_data.empty:
                # Return the requested price type of the closest date before
                return before_data.iloc[0][price_type]
            elif not after_data.empty:
                # Return the requested price type of the closest date after
                return after_data.iloc[0][price_type]
            
            return last_year_data.iloc[0][price_type]  # If no close dates are available, return the first available price

        except Exception as e:
            print(f"Error in fetching price data: {e}")
            return None