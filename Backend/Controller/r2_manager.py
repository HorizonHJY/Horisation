"""
r2_manager.py
Cloudflare R2 image upload/delete via boto3 (S3-compatible API).
Config loaded from Key/r2_config.json (gitignored).
"""

import os
import json
import uuid
import boto3
from botocore.exceptions import ClientError

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, 'Key', 'r2_config.json')

_config = None
_client = None


def _load_config():
    global _config
    if _config is None:
        with open(CONFIG_PATH, 'r') as f:
            _config = json.load(f)
    return _config


def _get_client():
    global _client
    if _client is None:
        cfg = _load_config()
        _client = boto3.client(
            's3',
            endpoint_url=f"https://{cfg['account_id']}.r2.cloudflarestorage.com",
            aws_access_key_id=cfg['access_key_id'],
            aws_secret_access_key=cfg['secret_access_key'],
            region_name='auto',
        )
    return _client


def upload_image(file_obj, original_filename: str) -> tuple[str, str]:
    """
    Upload a file-like object to R2.
    Returns (r2_key, public_url).
    r2_key is stored in DB for later deletion.
    """
    cfg    = _load_config()
    ext    = os.path.splitext(original_filename)[1].lower()
    r2_key = f"listings/{uuid.uuid4().hex}/{uuid.uuid4().hex}{ext}"

    content_type = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'

    _get_client().upload_fileobj(
        file_obj,
        cfg['bucket_name'],
        r2_key,
        ExtraArgs={'ContentType': content_type},
    )

    public_url = f"{cfg['public_url'].rstrip('/')}/{r2_key}"
    return r2_key, public_url


def delete_image(r2_key: str) -> bool:
    """Delete an object from R2 by its key. Returns True on success."""
    try:
        cfg = _load_config()
        _get_client().delete_object(Bucket=cfg['bucket_name'], Key=r2_key)
        return True
    except ClientError:
        return False
