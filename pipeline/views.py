from django.shortcuts import render

# Create your views here.

def home(request):
    """Home view for YouTube Video Pipeline dashboard."""
    return render(request, "pipeline/home.html")
