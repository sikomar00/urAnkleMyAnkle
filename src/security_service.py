"""Argon2id 비밀번호 검증과 AES-256-GCM 개인정보 암호화."""

import base64
import os
from pathlib import Path

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError, VerificationError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1, type=Type.ID)


def hash_password(password: str) -> str:
    # 시연 계정의 최초 비밀번호(1234)를 지원하기 위한 최소 길이이다.
    # 발표 후 실제 배포에서는 .env의 비밀번호를 12자 이상으로 바꾼다.
    if len(password) < 4:
        raise ValueError("관리자 비밀번호는 4자 이상이어야 합니다.")
    return _hasher.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError):
        return False


def _aes() -> AESGCM:
    raw = os.environ.get("DASHBOARD_AES_KEY", "")
    try:
        key = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise RuntimeError("DASHBOARD_AES_KEY 설정이 올바르지 않습니다.") from exc
    if len(key) != 32:
        raise RuntimeError("DASHBOARD_AES_KEY에는 32바이트 키의 Base64 값이 필요합니다.")
    return AESGCM(key)


def encrypt_personal_data(value: str) -> str:
    # 같은 키로 nonce를 재사용하지 않도록 매 암호화마다 새로 생성한다.
    nonce = os.urandom(12)
    ciphertext = _aes().encrypt(nonce, value.encode("utf-8"), None)
    return "v1:" + base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_personal_data(value: str) -> str:
    if not value.startswith("v1:"):
        raise ValueError("지원하지 않는 암호문 형식입니다.")
    payload = base64.b64decode(value[3:], validate=True)
    if len(payload) < 28:
        raise ValueError("암호문이 손상되었습니다.")
    return _aes().decrypt(payload[:12], payload[12:], None).decode("utf-8")
