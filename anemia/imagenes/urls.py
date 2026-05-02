from django.urls import path
from . import views

urlpatterns = [
    # Enlaces para realizar el procesamiento de imágenes
    path('crear_carpetas_iniciales/', views.crear_carpetas_iniciales, name='crear-carpetas-iniciales'),
    path('recortar_ojo/', views.ejecutar_recorte_ojo, name='recortar-ojo'),
    path('filtrar/', views.ejecutar_filtrado, name='filtrar-conjuntiva'),
    path('balancear/', views.ejecutar_balanceo, name='balancear-dataset'),
    path('segmentar/', views.ejecutar_segmentacion, name='segmentar-conjuntiva'),
    path('redimensionar/', views.ejecutar_redimensionamiento, name='redimensionar-imagenes'),
    path('aumentar/', views.ejecutar_aumentacion, name='aumentar-dataset'),
    
    # Nuevos endpoints
    path('ejecutar_todo/', views.ejecutar_todo, name='ejecutar-todo'),
    path('listar_imagenes/', views.listar_imagenes, name='listar-imagenes'),
    path('explorar/', views.explorar_carpetas, name='explorar-carpetas'),
    path('mover_archivo/', views.mover_archivo, name='mover-archivo'),
    path('preparar_dataset/', views.ejecutar_preparar_dataset, name='preparar-dataset'),
    path('prueba_rapida/', views.ejecutar_prueba_rapida, name='prueba-rapida'),
]
