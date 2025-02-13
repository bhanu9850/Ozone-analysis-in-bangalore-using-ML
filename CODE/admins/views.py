from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
from django.contrib.auth.models import User 




def adminhome(request):
    return render(request,'admin_home.html')

def adminlogin(request):
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'admin'

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            messages.success(request, "Welcome, Admin!")
            return redirect('/admins/userslist') 
        else:
            messages.error(request, "Invalid login credentials.")

    return render(request, 'admin_login.html')

    
def adminlogout(request):    
    logout(request)
    return redirect('/admins/admin-login')    
def userslist(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')  # Get the user ID from the form
        try:
            user = User.objects.get(id=user_id)  # Retrieve the user
            user.is_active = True  # Activate the user
            user.save()  # Save the changes
            messages.success(request, f"User {user.username} has been activated.")
        except User.DoesNotExist:
            messages.error(request, "User not found.")
        return redirect('userslist')  # Redirect to the same page after processing

    # Fetch all users for the GET request
    users = User.objects.all()
    context = {'users': users}
    return render(request, 'users_list.html', context) 