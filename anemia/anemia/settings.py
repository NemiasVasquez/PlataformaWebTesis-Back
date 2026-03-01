from pathlib import Path
import os
from dotenv import load_dotenv

# Cargar variables .env
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure-3xu=h15n7gfktaus$s!0qs-6j89x(ck(pcs29&6ec7v%+o6klo'
DEBUG = True

# Solo tu app personalizada (ej. 'modelo')
INSTALLED_APPS = [
    'corsheaders',
    'anemia',
    'modelo',
    'imagenes',
    
]

MIDDLEWARE = [    
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    ]  # Sin middleware de seguridad

ROOT_URLCONF = 'anemia.urls'

CORS_ALLOW_ALL_ORIGINS = True

TEMPLATES = []  # No se usan templates

WSGI_APPLICATION = 'anemia.wsgi.application'

# Base de datos mínima (no se usará, pero requerida por Django)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Localización básica
LANGUAGE_CODE = 'es-PE'
TIME_ZONE = 'America/Lima'
USE_I18N = False
USE_L10N = False
USE_TZ = False

# Media para recibir imágenes
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Auto field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'