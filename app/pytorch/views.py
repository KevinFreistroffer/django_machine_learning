from django.shortcuts import render

# Create your views here.

def index(request):
    """
    Main view for PyTorch application
    """
    context = {
        'title': 'PyTorch App',
        'welcome_message': 'Welcome to the PyTorch Application!'
    }
    return render(request, 'pytorch/index.html', context)
