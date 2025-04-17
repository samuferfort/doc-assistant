# Document Assistant

Este proyecto es un servicio de procesamiento de documentos que utiliza FastAPI para crear embeddings y procesar documentos PDF.

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

## Configuración del Entorno

### 1. Clonar el Repositorio

```bash
git clone https://github.com/YourUsername/doc-assistant.git
cd doc-assistant
```

### 2. Crear el Entorno Virtual

#### En Windows:

```powershell
# Crear el entorno virtual
python -m venv doc-assistant-env

# Activar el entorno virtual
# Si usas PowerShell:
.\venv\Scripts\Activate.ps1

# Si usas Command Prompt (cmd):
.\venv\Scripts\activate.bat
```

#### En macOS/Linux:

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate
```

### 3. Instalar Dependencias

Con el entorno virtual activado (deberías ver `(venv)` al inicio de tu línea de comando):

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
OPENAI_API_KEY=tu_api_key_de_openai
AWS_ACCESS_KEY_ID=tu_aws_access_key
AWS_SECRET_ACCESS_KEY=tu_aws_secret_key
AWS_S3_REGION=tu_region_aws
AWS_S3_BUCKET_NAME=nombre_de_tu_bucket
NEXTJS_CALLBACK_URL=url_de_callback
PYTHON_SERVICE_SECRET=tu_secreto
```

## Uso

### Iniciar el Servidor

```bash
uvicorn main:app --reload
```

El servidor estará disponible en `http://localhost:8000`

### Endpoints Disponibles

- `GET /health`: Verificar el estado del servicio
- `POST /process`: Procesar documentos

## Desactivar el Entorno Virtual

Cuando hayas terminado de trabajar en el proyecto, puedes desactivar el entorno virtual:

```bash
deactivate
```

## Solución de Problemas

### Error de Permisos en PowerShell

Si recibes un error al activar el entorno virtual en PowerShell, ejecuta:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Verificar Instalación

Para verificar que todas las dependencias están instaladas correctamente:

```bash
pip list
```

## Dependencias Principales

- FastAPI
- uvicorn
- langchain
- OpenAI
- boto3
- python-dotenv
- httpx
- pypdf
