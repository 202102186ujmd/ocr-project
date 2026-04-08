# OCR API con Paligemma

API de Reconocimiento Óptico de Caracteres (OCR) construida con FastAPI y el modelo Paligemma de Google.

## Estructura del Proyecto

```
ocr-project/
├── main.py                 # Aplicación FastAPI principal
├── config.py              # Configuración de la aplicación
├── requirements.txt       # Dependencias del proyecto
├── test_main.http         # Tests HTTP
├── models/
│   ├── __init__.py
│   └── ocr_model.py       # Integración del modelo Paligemma
├── schemas/
│   ├── __init__.py
│   └── schemas.py         # Modelos Pydantic para validación
├── routes/
│   ├── __init__.py
│   └── ocr_routes.py      # Endpoints de la API
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Funciones auxiliares
└── uploads/               # Carpeta para imágenes subidas
```

## Instalación

### Requisitos previos
- Python 3.8+
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar o descargar el proyecto**

2. **Crear un entorno virtual** (recomendado)
```bash
python -m venv venv
```

3. **Activar el entorno virtual**

En Windows:
```bash
venv\Scripts\activate
```

En Linux/Mac:
```bash
source venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## Uso

### Iniciar la aplicación

```bash
uvicorn main:app --reload
```

La API estará disponible en: `http://localhost:8000`

### Documentación interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints disponibles

### 1. Health Check
```
GET /api/ocr/health
```
Verifica el estado del servicio OCR.

### 2. Subir imagen
```
POST /api/ocr/upload
Content-Type: multipart/form-data

file: <archivo de imagen>
```
Respuesta:
```json
{
  "success": true,
  "filename": "20260408_120000_imagen.jpg",
  "file_path": "/path/to/uploads/20260408_120000_imagen.jpg",
  "message": "Archivo guardado exitosamente"
}
```

### 3. Extraer texto de imagen local
```
POST /api/ocr/extract?image_path=/path/to/image.jpg&prompt=¿Qué texto ves?
```

### 4. Extraer texto de URL
```
POST /api/ocr/extract-from-url?image_url=https://example.com/image.jpg&prompt=Extrae el texto
```

### 5. Root
```
GET /
```
Información general de la API.

### 6. Test
```
GET /hello/{name}
```
Endpoint de prueba.

## Configuración

Editar el archivo `config.py` para personalizar:

- `app_name`: Nombre de la aplicación
- `app_version`: Versión de la aplicación
- `upload_dir`: Directorio de uploads
- `paligemma_model`: Modelo a utilizar
- `device`: CPU o CUDA (GPU)
- `max_upload_size`: Tamaño máximo de archivo

## Modelos disponibles

- `google/paligemma-3b-pt-224` (por defecto)
- Otros modelos de Paligemma disponibles en HuggingFace

## Requisitos de GPU (opcional)

Para usar GPU (CUDA):

1. Instalar CUDA Toolkit
2. Cambiar `device` a `"cuda"` en `config.py`
3. Instalar versiones compatibles de torch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Estructura de respuestas

### OCRResponse
```json
{
  "success": true,
  "text": "Texto extraído de la imagen",
  "confidence": 0.85,
  "error": null
}
```

### HealthResponse
```json
{
  "status": "online",
  "version": "0.1.0",
  "message": "OCR API con Paligemma activa y funcionando"
}
```

## Solución de problemas

### Error: "Modelo no encontrado"
- Verificar conexión a internet (necesaria para descargar el modelo)
- Verificar que HuggingFace esté accesible

### Error: "CUDA out of memory"
- Cambiar a CPU en `config.py`
- Reducir tamaño de imagen
- Usar modelo más pequeño

### Lentitud al procesar
- Usar GPU en lugar de CPU
- Optimizar imagen (reducir resolución)
- Usar modelo quantizado

## Desarrollo

### Agregar nuevo endpoint
1. Crear función en `routes/ocr_routes.py`
2. Usar decoradores de FastAPI (`@router.get()`, `@router.post()`, etc.)
3. Definir modelos en `schemas/schemas.py`

### Agregar función auxiliar
1. Crear función en `utils/helpers.py`
2. Importar en el módulo que la necesite

## Licencia

Este proyecto utiliza:
- FastAPI (MIT)
- Paligemma (Apache 2.0)
- Transformers (Apache 2.0)

## Soporte

Para reportar problemas o sugerencias, crear un issue en el repositorio.

