from django.shortcuts import render

def home(request):
    """
    Main landing page view
    """
    context = {
        'title': 'Welcome',
        'description': 'Your AI Applications Hub'
    }
    return render(request, 'home.html', context) 