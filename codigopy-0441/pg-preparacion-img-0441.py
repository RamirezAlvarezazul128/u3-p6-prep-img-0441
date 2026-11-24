import csv
import os
import cv2
import numpy as np
from PIL import Image

## Ramirez Azul NC 0441
# 1. Configuración inicial (adaptada a la estructura 0441)
carpeta_original = "img-original-0441"
carpeta_preparadas = "img-preparadas-0441"
carpeta_resultados = "resultados-img-0441"

ANCHO_DESTINO = 224
ALTO_DESTINO = 224

# Crear carpetas de salida si no existen
os.makedirs(carpeta_preparadas, exist_ok=True)
os.makedirs(carpeta_resultados, exist_ok=True)

reporte = []
imagenes_procesadas = 0
imagenes_aprobadas = 0

print("🚀 Iniciando preparación de imágenes...\n")

# 2. Verificar que la carpeta de origen exista y recorrer sus archivos directamente
if not os.path.exists(carpeta_original):
  print(f"⚠️ La carpeta {carpeta_original} no existe.")
else:
  for nombre_archivo in sorted(os.listdir(carpeta_original)):
    # Filtrar solo archivos de imagen válidos
    if not nombre_archivo.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
      continue

    ruta_imagen = os.path.join(carpeta_original, nombre_archivo)

    # Ignorar si por alguna razón hay una subcarpeta
    if not os.path.isfile(ruta_imagen):
      continue

    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]

    # Determinar la categoría según el nombre del archivo (morado / naranja)
    if nombre_archivo.lower().startswith("morado"):
      categoria = "morado"
    elif nombre_archivo.lower().startswith("naranja"):
      categoria = "naranja"
    else:
      categoria = "sin_categoria"

    print(f"📸 Procesando: {nombre_archivo} (Categoría: {categoria})")

    # 3. Leer la imagen (con respaldo de Pillow por si es WebP/AVIF disfrazado de .jpg)
    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
      try:
        with Image.open(ruta_imagen) as img_pil:
          img_rgb = img_pil.convert("RGB")
          # Convertir formato RGB de Pillow al formato BGR que usa OpenCV
          imagen = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
      except Exception:
        imagen = None

    if imagen is None:
      print("   ❌ Error: No se pudo leer la imagen.\n")
      reporte.append([
          categoria,
          nombre_archivo,
          "ERROR",
          "-",
          "-",
          "-",
          "-",
          "-",
          "-",
          "REVISAR",
      ])
      continue

    imagenes_procesadas += 1
    alto_orig, ancho_orig, canales = imagen.shape
    print(f"   📏 Dimensiones originales: {ancho_orig} x {alto_orig}")

    # 4. Redimensionar la imagen
    imagen_redimensionada = cv2.resize(
        imagen, (ANCHO_DESTINO, ALTO_DESTINO), interpolation=cv2.INTER_AREA
    )
    print(f"   📐 Dimensiones nuevas: {ANCHO_DESTINO} x {ALTO_DESTINO}")

    # Guardar la imagen redimensionada como JPG directamente en img-preparadas-0441
    ruta_jpg = os.path.join(carpeta_preparadas, f"{nombre_sin_ext}.jpg")
    cv2.imwrite(ruta_jpg, imagen_redimensionada)

    # 5. Normalizar los valores de los píxeles (de 0-255 a 0.0-1.0)
    imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0
    val_min = float(np.min(imagen_normalizada))
    val_max = float(np.max(imagen_normalizada))
    print(f"   🔢 Rango normalizado: {val_min:.3f} a {val_max:.3f}")

    # Guardar los datos numéricos normalizados como archivo .npy
    ruta_npy = os.path.join(carpeta_preparadas, f"{nombre_sin_ext}.npy")
    np.save(ruta_npy, imagen_normalizada)

    # 6. Evaluar automáticamente el resultado
    dimensiones_correctas = (
        imagen_redimensionada.shape[0] == ALTO_DESTINO
        and imagen_redimensionada.shape[1] == ANCHO_DESTINO
    )
    rango_correcto = val_min >= 0.0 and val_max <= 1.0
    archivos_guardados = os.path.isfile(ruta_jpg) and os.path.isfile(ruta_npy)

    if dimensiones_correctas and rango_correcto and archivos_guardados:
      estado = "APROBADA"
      imagenes_aprobadas += 1
      print("   ✅ Evaluación: APROBADA\n")
    else:
      estado = "REVISAR"
      print("   ⚠️ Evaluación: REVISAR\n")

    # 7. Guardar datos en la lista para el reporte final
    reporte.append([
        categoria,
        nombre_archivo,
        "OK",
        ancho_orig,
        alto_orig,
        ANCHO_DESTINO,
        ALTO_DESTINO,
        round(val_min, 3),
        round(val_max, 3),])