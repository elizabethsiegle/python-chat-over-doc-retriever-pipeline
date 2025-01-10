import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the directory containing this script
current_dir = Path(__file__).parent.absolute()

# Look for .env file in current directory and parent directory
env_path = current_dir / '.env'
if not env_path.exists():
    env_path = current_dir.parent / '.env'

logger.debug(f"Looking for .env file at: {env_path}")

# Load environment variables
if env_path.exists():
    logger.debug(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    logger.error(f"No .env file found at {env_path}")
    raise FileNotFoundError(f"No .env file found at {env_path}")

# Get configuration values
VOYAGE_API_KEY = os.environ.get('VOYAGE_API_KEY')
CLOUDFLARE_ACCOUNT_ID = os.environ.get('CLOUDFLARE_ACCOUNT_ID')
CLOUDFLARE_AUTH_TOKEN = os.environ.get('CLOUDFLARE_AUTH_TOKEN')
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')

logger.debug(f"Loaded VOYAGE_API_KEY: {'Present' if VOYAGE_API_KEY else 'Missing'}")

if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in environment variables")
