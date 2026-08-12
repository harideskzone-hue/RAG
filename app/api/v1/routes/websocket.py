import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.presenters.websocket_presenter import WebSocketPresenter

router = APIRouter(prefix="/ws", tags=["websocket"])

@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            # Simulated streaming response
            await websocket.send_text(WebSocketPresenter.present_progress("MetadataAgent", 20))
            await asyncio.sleep(0.5)
            await websocket.send_text(WebSocketPresenter.present_progress("VideoAgent", 60))
            await asyncio.sleep(0.5)
            await websocket.send_text(WebSocketPresenter.present_token("I "))
            await asyncio.sleep(0.1)
            await websocket.send_text(WebSocketPresenter.present_token("found "))
            await asyncio.sleep(0.1)
            await websocket.send_text(WebSocketPresenter.present_token("the "))
            await asyncio.sleep(0.1)
            await websocket.send_text(WebSocketPresenter.present_token("person."))
            await asyncio.sleep(0.1)
            
            await websocket.send_text(WebSocketPresenter.present_completed({"status": "success"}))
            
    except WebSocketDisconnect:
        pass
