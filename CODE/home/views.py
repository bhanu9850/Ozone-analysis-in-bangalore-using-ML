from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import authenticate, login,logout
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from .models import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import joblib
import os
import time
from tqdm import tqdm




MODEL_FILE = "trained_lr_model.pkl"
 
def home(request):
    return render(request,"home.html")

def registerView(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        mobile = request.POST.get('mobile')
        email = request.POST.get('email')
        locality = request.POST.get('locality')
        address = request.POST.get('address')
        print(username,password,mobile,email,locality,address)
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
        else:
            user = User.objects.create_user(
                username=username, 
                password=password,
                email=email
            )
            user.is_active = False
            user.save()
            Profile.objects.create(
                user=user,
                mobile=mobile,
                locality=locality,
                address=address
            )
            messages.success(request, "Registration successful. Please log in.")
            return redirect('login-user')

    return render(request, 'register.html')   

def loginView(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        user = User.objects.filter(username=username).first()
        if user:
            if not user.is_active:  
                messages.error(request, "Your account is inactive. Please activate it in the admin panel.")
                return render(request, 'login.html')  
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome, {user.username}!")
                return redirect('home')
            else:
                messages.error(request, "Invalid login credentials.")
        else:
            messages.error(request, "Invalid login credentials.")
    return render(request, 'login.html')    
file_name = 'home/media/ozone prediction in banglore dataset.csv'  

def view_dataset(request): 
    df = pd.read_csv(file_name)
    columns_to_drop = ['Location', 'Month','Day']  
    df = df.drop(columns=columns_to_drop)
    top_100 = df.head(50)
    dataset_html = top_100.to_html(header=True, border=1, index=False)
    return render(request,'dataset.html',{'dataset_table': dataset_html})     



def train_model(request):
    progress = tqdm(total=100, desc="Training Progress", bar_format="{l_bar}{bar} [ time left: {remaining} ]")

    print("Reading the dataset...")
    df = pd.read_csv(file_name)
    time.sleep(1)  
    progress.update(10)
    print("Dataset loaded successfully!")

    print("Selecting features and target...")
    features = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO (ppb)', 'NO2 (ppb)',
                'NH3 (ppb)', 'SO2 (ppb)', 'CO (ppm)', 'Temp (°C)',
                'RH (%)', 'WS (m/s)', 'SR (W/m²)', 'BP (hPa)']
    target = 'Ozone (ppb)'
    X = df[features]
    y = df[target]
    progress.update(10)
    print("Features and target selected!")

    print("Splitting the dataset into training and testing sets...")
    time.sleep(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    progress.update(10)
    print("Dataset split completed!")

    print("Initializing the Linear Regression model...")
    time.sleep(1)
    lr_model = LinearRegression()
    progress.update(10)
    print("Model initialized!")

    print("Training the model...")
    time.sleep(2)  # Simulate training delay
    lr_model.fit(X_train, y_train)
    progress.update(20)
    print("Model training completed!")

    print("Predicting the test data...")
    time.sleep(1)
    lr_pred = lr_model.predict(X_test)
    progress.update(10)
    print("Predictions completed!")

    print("Calculating metrics...")
    time.sleep(1)
    rmse = np.sqrt(mean_absolute_error(y_test, lr_pred))
    max_value = df['Ozone (ppb)'].max()
    min_value = df['Ozone (ppb)'].min()
    normalized_rmse = rmse / (max_value - min_value)
    r2 = r2_score(y_test, lr_pred)
    mae = mean_absolute_error(y_test, lr_pred)
    mse = mean_squared_error(y_test, lr_pred)
    progress.update(20)
    print("Metrics calculated!")

    print("Saving the model...")
    joblib.dump(lr_model, MODEL_FILE)
    progress.update(10)
    print(f"Model saved successfully to {MODEL_FILE}!")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:500:5], label='True Values', marker='o')
    plt.plot(lr_pred[:500:5], label='Linear Regression Predictions', marker='x')
    plt.legend()
    plt.title("True vs Predicted Ozone Concentration - Linear Regression")
    plt.xlabel("Sample Index")
    plt.ylabel("Ozone Concentration (ppb)")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    image_base64 = base64.b64encode(image_png).decode('utf-8')
    
    print("\nTraining Summary:")
    print(f"R² Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Normalized RMSE: {normalized_rmse}")

    progress.close()

    return render(request, 'train_model.html', {
        'r2_score': r2,
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'root_mean_squared_error': rmse,
        'normalized_rmse': normalized_rmse,
        'plot_image': image_base64,
    })


def userlogout(request):
    logout(request)
    return redirect('login-user')

def prediction(request):
    if request.method == 'POST':
        try:
            lr_model = joblib.load(MODEL_FILE)
        except FileNotFoundError:
            return HttpResponse("Model file not found. Please train the model first.", status=500)
        except Exception as e:
            return HttpResponse(f"An error occurred: {e}", status=500)
        input_values = [
            [
                float(request.POST.get('PM2.5')),
                float(request.POST.get('PM10')),
                float(request.POST.get('NO')),
                float(request.POST.get('NO2')),
                float(request.POST.get('NH3')),
                float(request.POST.get('SO2')),
                float(request.POST.get('CO')),
                float(request.POST.get('Temp')),
                float(request.POST.get('RH')),
                float(request.POST.get('WS')),
                float(request.POST.get('SR')),
                float(request.POST.get('BP'))
            ]
        ]
        print(input_values)
        prediction = lr_model.predict(input_values)
        predicted_ozone = prediction[0]
        print(prediction)
        context = {
            'predicted_ozone': f"{predicted_ozone:.2f}",
            'input_values': input_values[0]
        }
        print(context)
        return render(request, 'ozone_form.html', context)
    return render(request, 'ozone_form.html')
