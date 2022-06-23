# Segmentación automática de glándulas en imágenes histológicas de cáncer colorrectal con tinción Hematoxilina-Eosina
## Trabajo para la asignatura Imagen Médica del Master Universitario en Visión Artificial, URJC

Autores: Katherine Coutinho García, Paula Sánchez Tirado, David María Arribas

### Ficheros de la práctica

- dataset_generator.py: genera el dataset para entrenamiento y test, mediante normalización de color, división en parches de 255 por 255 y aumentado de datos.
- model_unet.py: primer modelo Unet reducido, empleado en las pruebas.
- model_unet_v2.py: segundo modelo Unet empleado en las pruebas.
- model_unet_v3.py: modelo Unet definitivo.
- lighning_model.py: modelo *Pytorch Lightning* empleado por los tres modelos Unet.
- train.py: script para training.
- test.py: script para testing y generación de resultados.
- predict.py: script para inferencia/predicción sobre los parches 256x256.
- predict_original.py: script para predicción sobre las imágenes completas.

También se incluye [smooth_tiled_predictions.py](https://github.com/Vooban/Smoothly-Blend-Image-Patches) para reunir los parches una vez realizada la predicción. Gracias a 
[Guillaume Chevalier](https://github.com/guillaume-chevalier) por esta gran contribución.