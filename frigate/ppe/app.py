import os
import cv2
import base64
import asyncio
import threading
from fastapi import FastAPI, WebSocket, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from util_classes import DetectorServer

# ---------------- Config ----------------
JPEG_QUALITY = 70
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "static", "index.html")
MODEL_PATH = os.path.join(BASE_DIR, "best_yolo11.pt")

# create folders if missing
os.makedirs("saved_images", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------- FastAPI App ----------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

detector = None
detector_lock = threading.Lock()
ws_clients = set()   # allow multiple clients optionally

@app.get("/")
def index():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/start")
def start(source: str = Query("webcam")):
    global detector
    with detector_lock:
        if detector:
            try:
                detector.stop()
            except:
                pass
            detector = None
        src = 0 if source == "webcam" else source
        detector = DetectorServer(src=src, model_path=MODEL_PATH)
        detector.start()
    return {"status":"started", "source": source}

@app.get("/stop")
def stop():
    global detector
    with detector_lock:
        if detector:
            try:
                detector.stop()
            except:
                pass
            detector = None
    return {"status":"stopped"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    path = os.path.join("static", file.filename)
    with open(path,"wb") as f:
        f.write(contents)
    return {"path": path}

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    print("[WS] client connected")
    ws_clients.add(ws)
    try:
        while True:
            # keep reading frames from detector and send to this client
            if detector is None:
                await asyncio.sleep(0.2)
                continue
            item = detector.get_frame(timeout=1.0)
            if item is None:
                await asyncio.sleep(0.02)
                continue
            frame, dets, tracks, stats = item
            try:
                _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
                payload = {
                    "type":"frame",
                    "frame": b64,
                    "detections": dets,
                    "tracks": tracks,
                    "stats": stats
                }
                await ws.send_json(payload)
            except Exception as e:
                print("[WS] send error:", e)
                break
            await asyncio.sleep(0.01)
    except Exception as e:
        print("[WS] connection error:", e)
    finally:
        try:
            await ws.close()
        except:
            pass
        if ws in ws_clients:
            ws_clients.remove(ws)
        print("[WS] client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
