from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.deps import pool, settings

router = APIRouter()


async def check_database() -> bool:
    try:
        await pool.execute("SELECT 1")
        return True
    except Exception:
        return False


async def check_redis() -> bool:
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


@router.get("/health")
async def health() -> JSONResponse:
    db_ok = await check_database()
    redis_ok = await check_redis()
    status = 200 if db_ok and redis_ok else 503
    return JSONResponse(content={"db": db_ok, "redis": redis_ok}, status_code=status)
