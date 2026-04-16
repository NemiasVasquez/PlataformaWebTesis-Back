from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def crear_modelo():
    base = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))  # solo 3 canales
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    salida = Dense(1, activation='sigmoid')(x)
    modelo = Model(inputs=base.input, outputs=salida)
    modelo.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return modelo
