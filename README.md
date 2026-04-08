# OCR API con Paligemma

API de Reconocimiento Óptico de Caracteres (OCR) construida con **FastAPI** y el modelo **Paligemma** de Google, optimizada para procesar documentos de identidad salvadoreños (DUI y licencias de conducir).

## Estructura del Proyecto

```
ocr-project/
├── main.py                 # Aplicación FastAPI principal
├── config.py              # Configuración de la aplicación
├── requirements.txt       # Dependencias del proyecto
├── .env.example           # Variables de entorno de ejemplo
├── test_main.http         # Tests HTTP (JetBrains HTTP Client)
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
│   └── helpers.py         # Funciones auxiliares y validadores
└── uploads/               # Carpeta temporal para imágenes subidas
```

## Instalación

### Requisitos previos
- Python 3.8+
- pip (gestor de paquetes de Python)
- GPU con al menos 6 GB de VRAM (recomendado para float16)

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

5. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env según tu entorno
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

### Health Check
```
GET /api/ocr/health
```
Verifica el estado del servicio OCR.

### Subir imagen
```
POST /api/ocr/upload
Content-Type: multipart/form-data

file: <archivo de imagen>
```

### Extraer texto de imagen local
```
POST /api/ocr/extract?image_path=/path/to/image.jpg&prompt=¿Qué texto ves?
```

### Extraer texto de URL
```
POST /api/ocr/extract-from-url?image_url=https://example.com/image.jpg
```

### Extraer campos de DUI ⭐
```
POST /api/ocr/extract-dui
Content-Type: multipart/form-data

file: <imagen del DUI>
```
Respuesta:
```json
{
  "apellidos": "GARCIA LOPEZ",
  "nombres": "JUAN CARLOS",
  "genero": "M",
  "fecha_nacimiento": "1990-05-15",
  "lugar_nacimiento": "San Salvador",
  "numero_dui": "01234567-8",
  "raw_extraction": "...",
  "processing_time": 1.234,
  "valid": true
}
```

### Extraer campos de Licencia de Conducir ⭐
```
POST /api/ocr/extract-license
Content-Type: multipart/form-data

file: <imagen de la licencia>
```
Respuesta:
```json
{
  "nombre": "Juan Carlos Garcia",
  "numero_licencia": "ABC123456",
  "dui": "01234567-8",
  "clase_categoria": "D",
  "fecha_vencimiento": "2026-12-31",
  "genero": "M",
  "raw_extraction": "...",
  "processing_time": 1.456,
  "valid": true
}
```

### Validar integridad de documento ⭐
```
POST /api/ocr/validate-document?document_type=dui
Content-Type: multipart/form-data

file: <imagen del documento>
```
Respuesta:
```json
{
  "document_type": "dui",
  "is_valid": true,
  "errors": [],
  "warnings": ["Género no encontrado"],
  "fields": { ... }
}
```

### Procesamiento por lotes ⭐
```
POST /api/ocr/batch-process
Content-Type: multipart/form-data

files: <imagen 1>
files: <imagen 2>
...
```
Respuesta:
```json
{
  "total": 3,
  "successful": 2,
  "failed": 1,
  "results": [ ... ],
  "processing_time": 4.567
}
```

## Configuración

Editar el archivo `.env` (basado en `.env.example`) o directamente `config.py`:

| Variable | Descripción | Default |
|---|---|---|
| `PALIGEMMA_MODEL` | HuggingFace model ID | `google/paligemma-3b-pt-448` |
| `DEVICE` | `cuda` o `cpu` | `cuda` |
| `DTYPE` | `float16` (GPU) o `float32` (CPU) | `float16` |
| `USE_INT8` | Cuantización int8 para reducir VRAM | `false` |
| `UPLOAD_DIR` | Directorio de uploads temporales | `./uploads` |
| `MAX_UPLOAD_SIZE` | Tamaño máximo de archivo (bytes) | `10485760` |
| `MAX_NEW_TOKENS` | Tokens máximos a generar | `512` |

## Optimización de GPU

Para usar GPU con float16 (~6 GB VRAM):
```bash
# En .env
DEVICE=cuda
DTYPE=float16
```

Para cuantización int8 (~4 GB VRAM, requiere bitsandbytes):
```bash
USE_INT8=true
```

Para CPU (sin GPU):
```bash
DEVICE=cpu
DTYPE=float32
```

## Especificaciones de documentos soportados

### DUI de El Salvador
- **Campos extraídos**: Apellidos, Nombres, Género, Fecha de Nacimiento, Lugar de Nacimiento, Número Único de Identidad
- **Validación**: Número DUI con formato `XXXXXXXX-X` (9 dígitos)

### Licencia de Conducir de El Salvador
- **Campos extraídos**: Nombre completo, Número de Licencia, DUI/ID, Clase/Categoría, Fecha de Vencimiento, Género
- **Validación**: Fecha de vencimiento en rango 2020-2035, número de licencia con mínimo 4 caracteres

## Solución de problemas

### Error: "Modelo no encontrado"
- Verificar conexión a internet (necesaria para descargar el modelo la primera vez)
- Verificar que HuggingFace esté accesible

### Error: "CUDA out of memory"
- Activar cuantización int8: `USE_INT8=true`
- Cambiar a CPU: `DEVICE=cpu`, `DTYPE=float32`
- Reducir resolución de imagen

### Lentitud al procesar
- Usar GPU en lugar de CPU
- Optimizar imagen (reducir resolución)
- Usar cuantización int8

## Desarrollo

### Agregar nuevo endpoint
1. Crear función en `routes/ocr_routes.py`
2. Definir schema en `schemas/schemas.py`
3. Agregar tests en `test_main.http`

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

