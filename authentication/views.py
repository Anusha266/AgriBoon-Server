from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CustomUser
from .serializers import CustomUserSerializer

class RegisterView(APIView):
    def post(self, request):
        data = request.data
        print(data)
        try:
            serializer = CustomUserSerializer(data=data)
            if serializer.is_valid():
                serializer.save()
                return Response(
                    {
                        'message': 'User registered successfully',
                        'data':{
                            'data': serializer.data
                        }
                    }, 
                    status=status.HTTP_201_CREATED
                )

            return Response(
                {
                    'message': 'User registration failed',
                    'data':{
                        'error': serializer.errors
                    }
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {
                    'message': 'An error occurred during registration',
                    'data':{
                        'error': str(e)
                    }
                },
                status=status.HTTP_400_BAD_REQUEST
            )