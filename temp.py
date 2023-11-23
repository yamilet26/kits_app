# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
#from fastapi import FastAPI, File, UploadFile, HTTPException
#if os.name != 'nt':  # Solo importa fcntl en sistemas no-Windows
#    from gunicorn import util
#from fastapi import FastAPI
# from gunicorn.app.base import BaseApplication
# from uvicorn.workers import UvicornWorker
#app = FastAPI()
#app = Flask(name)
app = Flask(__name__)
#app = FastAPI()

# Configurar el modelo YOLO fuera de la función de detección
modelo_pesos = os.path.abspath('yolov3-spp.weights')
modelo_configuracion = os.path.abspath('yolov3-spp.cfg')
net = cv2.dnn.readNet(modelo_pesos, modelo_configuracion)
distancia_fija_cm = 170.0

def draw_height_on_image(image, altura):
    cv2.putText(image, f'Altura Persona: {altura:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Recibe la imagen desde la solicitud POST
            file = request.files['image']
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({'error': 'No se pudo decodificar la imagen'}), 400

            # Configurar la red neuronal para la detección de objetos
            layer_names = net.getUnconnectedOutLayersNames()

            # Obtener las dimensiones de la imagen
            height, width, _ = image.shape

            # Convertir la imagen a un blob para alimentar a la red neuronal
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)

            # Obtener las salidas de la red neuronal
            outs = net.forward(layer_names)

            # Inicializar listas para almacenar alturas y cuadros delimitadores
            alturas_personas = []
            cuadros_delimitadores = []

            # Iterar sobre las salidas de la red neuronal
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filtrar detecciones de personas con una confianza superior al 70%
                    if confidence > 0.7 and class_id == 0:
                        # Obtener las coordenadas de la caja delimitadora
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Dibujar el cuadro delimitador
                        x, y = int(center_x - w / 2), int(center_y - h / 2)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Calcular la altura relativa de la persona en centímetros
                        altura_persona_relativa_cm = (h / height) * distancia_fija_cm

                        # Almacenar la altura y el cuadro delimitador
                        alturas_personas.append(altura_persona_relativa_cm)
                        cuadros_delimitadores.append((x, y, x + w, y + h))

            # Dibujar la altura en la imagen
            if alturas_personas:
                image = draw_height_on_image(image, max(alturas_personas))

            # Devuelve las alturas como respuesta JSON
            response = {'alturas_personas': alturas_personas}
            print("Response:", response)  # Mensaje de depuración
            return jsonify(response)

        except Exception as e:
            print("Error:", str(e))  # Mensaje de depuración
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

#if name == 'main':
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# class FastAPIApplication(BaseApplication):
#     def __init__(self, app, options=None):
#         self.options = options or {}
#         self.application = app
#         super().__init__()

#     def load_config(self):
#         config = {
#             key: value for key, value in self.options.items()
#             if key in self.cfg.settings and value is not None
#         }
#         for key, value in config.items():
#             self.cfg.set(key.lower(), value)

#     def load(self):
#         return self.application

# if __name__ == '__main__':
#     options = {
#         'bind': '0.0.0.0:5000',  # Puedes cambiar el puerto si es necesario
#         'workers': 4,  # Puedes ajustar la cantidad de workers según tu necesidad
#         'worker_class': 'uvicorn.workers.UvicornWorker',
#         'reload': True,  # Activa el recargue automático
#     }
#     FastAPIApplication(app, options).run()
