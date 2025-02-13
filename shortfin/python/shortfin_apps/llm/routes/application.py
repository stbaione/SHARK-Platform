from fastapi import APIRouter, Response

application_router = APIRouter()


@application_router.get("/health")
async def health() -> Response:
    return Response(status_code=200)
