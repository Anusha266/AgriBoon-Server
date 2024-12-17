from rest_framework import serializers
from .models import CustomUser
from phonenumber_field.serializerfields import PhoneNumberField

class CustomUserSerializer(serializers.ModelSerializer):
    phone_number = PhoneNumberField()
    password = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser
        fields = ('phone_number', 'password', 'first_name', 'state', 
                 'district', 'mandal', 'village')
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def create(self, validated_data):
        user = CustomUser.objects.create_user(
            phone_number=validated_data['phone_number'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            state=validated_data.get('state', ''),
            district=validated_data.get('district', ''),
            mandal=validated_data.get('mandal', ''),
            village=validated_data.get('village', '')
        )
        return user
    
    
    