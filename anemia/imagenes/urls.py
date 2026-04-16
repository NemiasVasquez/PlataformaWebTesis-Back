from django.urls import path
from . import views

urlpatterns = [
    #Enlaces para realizar el procesamiento de imágenes
    path('crear_carpetas_iniciales/', views.crear_carpetas_iniciales, name='crear-carpetas-iniciales'),
    path('filtrar/', views.ejecutar_filtrado, name='filtrar-conjuntiva'),
    path('balancear/', views.ejecutar_balanceo, name='balancear-dataset'),
    path('segmentar/', views.ejecutar_segmentacion, name='segmentar-conjuntiva'),
    path('redimensionar/', views.ejecutar_redimensionamiento, name='redimensionar-imagenes'),

    # enlace para entrenamiento
]
