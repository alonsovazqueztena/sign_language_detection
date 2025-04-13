from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from . import api_utils  # Import the utility function defined earlier

@api_view(['POST'])
def detect_sign_language(request):
    """
    API endpoint that accepts an image upload and returns sign language detection results.
    """
    # Check if the image file is present in the request
    if 'image' not in request.FILES:
        return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)
    
    image_file = request.FILES['image']
    
    try:
        # Read the file bytes and process the image
        file_bytes = image_file.read()
        detections = api_utils.process_frame_from_buffer(file_bytes)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response({'detections': detections}, status=status.HTTP_200_OK)
