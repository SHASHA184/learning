import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer

security = HTTPBearer()


def create_jwt_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=1),  # Token expires in 1 hour
    }
    token = jwt.encode(payload, "your_secret_key", algorithm="HS256")
    return token


def verify_jwt_token(token: str = Depends(security)) -> int:
    try:
        payload = jwt.decode(token.credentials, "your_secret_key", algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
