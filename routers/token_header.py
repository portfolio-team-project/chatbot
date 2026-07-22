import os

from fastapi import Header, HTTPException, status


def get_token_header_name() -> str:
    token_header_name = os.environ.get("TOKEN_HEADER_NAME")
    if not token_header_name:
        raise RuntimeError("TOKEN_HEADER_NAME is not set")
    return token_header_name


def get_access_token() -> str:
    access_token = os.environ.get("ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("ACCESS_TOKEN is not set")
    return access_token


TOKEN_HEADER_NAME = get_token_header_name()
ACCESS_TOKEN = get_access_token()


def verify_token_header(access_token: str | None = Header(default=None, alias=TOKEN_HEADER_NAME)) -> None:
    if access_token != ACCESS_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")
