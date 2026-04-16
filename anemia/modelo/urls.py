from django.urls import path
from . import views

urlpatterns = [
    #Enlaces para realizar el entrenamient del modelo.
    path('entrenar_modelo/', views.entrenar_modelo_nfnet, name='entrenar-modelo'),
    path('evaluar_imagen_anemia/', views.evaluar_imagen_anemia, name='evaluar-imagen-anemia'),
]
