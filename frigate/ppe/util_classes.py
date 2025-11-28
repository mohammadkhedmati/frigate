import cv2
import time
import queue
import threading
import numpy as np
from datetime import datetime, timedelta
import helper
import os
# ---------------- Config ----------------
OUT_QUEUE_MAX = 2
CONF_TH = 0.35
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_yolo11.pt")
# ---------------- SimpleTracker (lightweight) ----------------
class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 1
        self.tracks = {}  # id -> {box, feat, first_seen, last_seen, lost_count, history}
        self.max_lost = max_lost

    def update(self, detections, frame_rgb, ts=None):
        if ts is None: ts = time.time()
        person_dets = []
        other_dets = []
        for det in detections:
            box,label,conf = det
            if 'person' in label:
                person_dets.append((box,conf))
            else:
                other_dets.append((box,label,conf))

        assigned = {}
        used = set()
        det_feats = []
        for (box,conf) in person_dets:
            crop = helper.crop_rgb(frame_rgb, box)
            feat = helper.color_hist_feature(crop) if crop is not None else None
            det_feats.append(feat)

        track_ids = list(self.tracks.keys())
        if len(track_ids)>0 and len(person_dets)>0:
            scores = np.zeros((len(track_ids), len(person_dets)), dtype=np.float32)
            for i,tid in enumerate(track_ids):
                tr = self.tracks[tid]
                for j,(box,conf) in enumerate(person_dets):
                    iou_score = helper.iou(tr["box"], box)
                    app_score = helper.cosine_sim(tr.get("feat"), det_feats[j]) if tr.get("feat") is not None and det_feats[j] is not None else 0.0
                    scores[i,j] = 0.65 * iou_score + 0.35 * app_score
            while True:
                i,j = np.unravel_index(np.argmax(scores), scores.shape)
                if scores[i,j] <= 0.1:
                    break
                tid = track_ids[i]
                assigned[tid] = j
                used.add(j)
                scores[i,:] = -1
                scores[:,j] = -1

        # update assigned
        for tid,j in assigned.items():
            box,conf = person_dets[j]
            feat = det_feats[j]
            tr = self.tracks[tid]
            tr["box"] = box
            if feat is not None: tr["feat"] = feat
            tr["last_seen"] = ts
            tr["lost_count"] = 0
            tr["history"].append((ts,box))

        # create new tracks
        for idx,(box_conf) in enumerate(person_dets):
            if idx in used: continue
            box,conf = box_conf
            feat = det_feats[idx]
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {
                "id": tid,
                "box": box,
                "feat": feat,
                "first_seen": ts,
                "last_seen": ts,
                "lost_count": 0,
                "history": [(ts,box)]
            }

        # increment lost
        to_delete = []
        for tid,tr in list(self.tracks.items()):
            if ts - tr["last_seen"] > 0.001 and tid not in assigned:
                tr["lost_count"] += 1
            if tr["lost_count"] > self.max_lost:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # compose outputs with PPE status by overlap
        outputs = []
        for tid,tr in self.tracks.items():
            box = tr["box"]
            has_helmet = False; has_vest = False
            for other in other_dets:
                obox, olabel, oconf = other
                cx,cy = helper.center_point(obox)
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    if 'helmet' in olabel or 'hardhat' in olabel: has_helmet = True
                    if 'vest' in olabel or 'safety' in olabel: has_vest = True
                else:
                    if helper.iou(box, obox) > 0.03:
                        if 'helmet' in olabel or 'hardhat' in olabel: has_helmet = True
                        if 'vest' in olabel or 'safety' in olabel: has_vest = True
            outputs.append({
                "id": tid,
                "box": box,
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "first_seen": tr["first_seen"],
                "last_seen": tr["last_seen"],
                "history": tr["history"]
            })
        return outputs

# ---------------- DetectorServer (reader + processor) ----------------
class DetectorServer:
    def __init__(self, src=0, model_path=MODEL_PATH):
        print(os.getcwd())
        print("[INFO] Loading model from:", model_path)
        self.model = helper.safe_load_model(model_path)
        self.src = src
        self.capture = None
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.out_queue = queue.Queue(maxsize=OUT_QUEUE_MAX)
        self.tracker = SimpleTracker(max_lost=30)
        self.events = []
        self._reader_thread = None
        self._proc_thread = None

    def start(self):
        if isinstance(self.src, str) and self.src.isdigit():
            self.src = int(self.src)
        self.capture = cv2.VideoCapture(self.src)
        if not self.capture.isOpened():
            raise RuntimeError("Cannot open video source: "+str(self.src))
        self.running = True
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._proc_thread = threading.Thread(target=self._processor, daemon=True)
        self._reader_thread.start()
        self._proc_thread.start()
        print("[INFO] DetectorServer started")

    def stop(self):
        self.running = False
        try:
            if self.capture:
                self.capture.release()
                self.capture = None
        except:
            pass
        with self.frame_lock:
            self.frame = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self._proc_thread and self._proc_thread.is_alive():
            self._proc_thread.join(timeout=1.0)
        try:
            while not self.out_queue.empty():
                self.out_queue.get_nowait()
        except:
            pass
        print("[INFO] DetectorServer stopped")

    def _reader(self):
        while self.running and self.capture:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.frame_lock:
                self.frame = frame
            time.sleep(0.001)

    def _processor(self):
        # Inference + tracking + annotate + push to out_queue
        # label matching: use model.names mapping; fallback to substring matching
        names_map = {i: name.lower() for i, name in self.model.names.items()} if hasattr(self.model, "names") else {}
        while self.running:
            with self.frame_lock:
                if self.frame is None:
                    frame_local = None
                else:
                    frame_local = self.frame.copy()
                    self.frame = None
            if frame_local is None:
                time.sleep(0.005); continue

            t0 = time.time()
            try:
                results = self.model.predict(source=frame_local, conf=CONF_TH, imgsz=640, verbose=False)
                detections = []
                # Collect detections: (box,label,conf)
                if len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None:
                        for box in boxes:
                            # robust attribute access
                            try:
                                conf = float(box.conf[0])
                            except Exception:
                                conf = float(getattr(box, "conf", 0.0))
                            try:
                                cls_idx = int(box.cls[0])
                            except Exception:
                                cls_idx = int(getattr(box, "cls", 0))
                            # label resolution
                            label = names_map.get(cls_idx, str(cls_idx)).lower()
                            try:
                                xy = box.xyxy[0]
                            except Exception:
                                xy = box.xyxy
                            x1,y1,x2,y2 = map(int, xy)
                            detections.append(((x1,y1,x2,y2), label, conf))

                # produce tracks
                frame_rgb = cv2.cvtColor(frame_local, cv2.COLOR_BGR2RGB)
                tracks = self.tracker.update(detections, frame_rgb, ts=time.time())

                # compute counts and annotate
                person_count = 0; helmet_count = 0; vest_count = 0
                annotated = frame_local
                # also produce flat detection list for precise counts (each detection has label/conf)
                flat_dets = []
                for (box,label,conf) in detections:
                    flat_dets.append({"box":box, "label":label, "conf":conf})
                    if 'person' in label: person_count += 1
                    if 'helmet' in label or 'hardhat' in label: helmet_count += 1
                    if 'vest' in label or 'safety' in label: vest_count += 1

                # annotate detections (boxes + confidence)
                for d in flat_dets:
                    x1,y1,x2,y2 = map(int, d["box"])
                    lab = d["label"]
                    conf = d["conf"]
                    if 'person' in lab:
                        col = (0,200,0)
                    elif 'helmet' in lab or 'hardhat' in lab:
                        col = (0,165,255)
                    elif 'vest' in lab or 'safety' in lab:
                        col = (255,165,0)
                    else:
                        col = (180,180,180)
                    helper.draw_box_cv(annotated, d["box"], text=f"{lab} {conf:.2f}", color=col)

                # annotate tracks with ID and PPE summary
                for tr in tracks:
                    tid = tr["id"]
                    box = tr["box"]
                    label_text = f"ID:{tid}"
                    if tr["has_helmet"]: label_text += " H"
                    if tr["has_vest"]: label_text += " V"
                    helper.draw_box_cv(annotated, box, text=label_text, color=(0,120,255))

                # prepare payload data
                tracks_out = []
                for d in flat_dets:
                    tracks_out.append({
                        "box": [int(d["box"][0]), int(d["box"][1]), int(d["box"][2]), int(d["box"][3])],
                        "label": d["label"],
                        "conf": float(d["conf"])
                    })
                # also expose tracker outputs
                tracker_out = []
                for t in tracks:
                    tracker_out.append({
                        "id": int(t["id"]),
                        "box": [int(t["box"][0]), int(t["box"][1]), int(t["box"][2]), int(t["box"][3])],
                        "has_helmet": bool(t["has_helmet"]),
                        "has_vest": bool(t["has_vest"]),
                        "first_seen": t["first_seen"],
                        "last_seen": t["last_seen"]
                    })

                # push annotated frame to out_queue (drop-oldest strategy)
                try:
                    if self.out_queue.full():
                        try:
                            _ = self.out_queue.get_nowait()
                        except:
                            pass
                    self.out_queue.put_nowait((annotated, tracks_out, tracker_out, {"person": person_count, "helmet": helmet_count, "vest": vest_count}))
                except Exception as e:
                    print("[WARN] out_queue put failed:", e)

            except Exception as e:
                print("[ERROR] processing frame:", e)

            t1 = time.time()
            proc_t = (t1 - t0)
            if proc_t < 0.01:
                time.sleep(0.005)

    def get_frame(self, timeout=1.0):
        try:
            return self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _log_event(self, etype, track_id, info):
        ts = datetime.now(datetime.timezone.utc).isoformat()
        ev = {"ts": ts, "type": etype, "track_id": track_id, "info": info}
        recent_same = [e for e in self.events if e["type"]==etype and e["track_id"]==track_id and (datetime.fromisoformat(e["ts"])> datetime.utcnow() - timedelta(seconds=10))]
        if len(recent_same)==0:
            self.events.append(ev)
            print("[EVENT]", ev)
