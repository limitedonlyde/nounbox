"""Шифрование секретов и маскирование token_id."""

import stat

import pytest
from cryptography.fernet import Fernet

from app import crypto


def test_key_file_created_with_0600(key_path):
    crypto.encrypt_secret("as-secret")

    assert key_path.exists()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    Fernet(key_path.read_bytes().strip())  # валидный ключ Fernet


def test_key_is_reused_between_calls(key_path):
    token = crypto.encrypt_secret("as-secret")
    key = key_path.read_bytes()

    crypto.reset_key_cache()
    assert crypto.decrypt_secret(token) == "as-secret"
    assert key_path.read_bytes() == key


def test_encrypt_roundtrip_hides_plaintext(key_path):
    secret = "as-1234567890abcdef"
    token = crypto.encrypt_secret(secret)

    assert secret not in token
    assert token != secret
    assert crypto.decrypt_secret(token) == secret


def test_env_key_overrides_file(tmp_path, monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(crypto.settings, "settings_encryption_key", key)
    monkeypatch.setattr(crypto.settings, "settings_key_path", str(tmp_path / "unused.key"))
    crypto.reset_key_cache()

    token = crypto.encrypt_secret("as-secret")

    assert not (tmp_path / "unused.key").exists()
    assert Fernet(key.encode()).decrypt(token.encode()).decode() == "as-secret"
    crypto.reset_key_cache()


def test_decrypt_with_other_key_raises(key_path, monkeypatch):
    token = crypto.encrypt_secret("as-secret")

    monkeypatch.setattr(
        crypto.settings, "settings_encryption_key", Fernet.generate_key().decode()
    )
    crypto.reset_key_cache()
    with pytest.raises(crypto.SecretDecryptError):
        crypto.decrypt_secret(token)


def test_bad_key_raises_secret_key_error(key_path):
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("not-a-fernet-key")
    crypto.reset_key_cache()

    with pytest.raises(crypto.SecretKeyError):
        crypto.encrypt_secret("as-secret")


@pytest.mark.parametrize(
    ("token_id", "expected"),
    [
        ("ak-1234567890abcdef", "ak-1234...cdef"),
        (None, None),
        ("", None),
    ],
)
def test_mask_token_id(token_id, expected):
    assert crypto.mask_token_id(token_id) == expected


def test_mask_token_id_short_does_not_reveal_all():
    masked = crypto.mask_token_id("ak-secret42")

    assert masked == "ak-..."
    assert "secret42" not in masked
