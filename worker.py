# worker.py — SAM/MatAnyone2 inference worker (runs OUTSIDE Nuke).
#
# This worker is the ONLY place torch is imported. Endpoint-security (EDR)
# tooling in some Windows environments blocks torch's c10.dll initialization
# (WinError 1114) in ANY descendant of Nuke.exe — proven by bisect: identical
# env and module lists, loads fine from PowerShell or a WMI-detached process,
# dies in every child of Nuke. So there are two transports:
#
#   --serve <rendezvous>   Detached socket mode (Windows). The client spawns
#       this via WMI (parent: WmiPrvSE.exe — outside Nuke's tree), we bind a
#       localhost port and advertise {port, token, pid} in the rendezvous
#       file. JSON-lines over the socket; all logs go to <rendezvous>.log.
#       Idle self-shutdown prevents orphaned workers.
#
#   (no args)              Piped stdio mode (POSIX / fallback): stdin one
#       JSON request per line, stdout one JSON response per line, logs on
#       stderr.
#
# Requests:
#   {"cmd": "ping"}                     -> {"ok": true, "torch": "...", "cuda": true}
#   {"cmd": "infer", ...}               -> {"ok": true, "message": "..."} + mask PNG
#   {"cmd": "clear"}                    -> {"ok": true}   (free models / VRAM)
#   {"cmd": "quit"}                     -> {"ok": true}   (process exits)
#
# The heavy mask math is NOT duplicated here: the nuke-free helpers in
# inference.py (finalize_mask, load_text_model, select_detection_by_point,
# _resolve_autocast_ctx) are imported and reused.

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make the package importable when run as a plain script from the venv.
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR.parent))

# ── Transport / logging setup (before any heavy import so failures are
#    captured) ──
_RDV = None
if "--serve" in sys.argv:
    # Detached mode: no console, no pipes — all output goes to the log file.
    _RDV = sys.argv[sys.argv.index("--serve") + 1]
    _logf = open(_RDV + ".log", "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _logf
    sys.stderr = _logf
    _PROTO = None
else:
    # stdio mode: stdout is reserved for protocol JSON. Reroute every print()
    # (ours and the ML libraries') to stderr so it can't corrupt the channel.
    _PROTO = sys.stdout
    sys.stdout = sys.stderr


def _log(msg):
    print(f"[H2 worker] {msg}", flush=True)


# IMPORT ORDER MATTERS: torch must load BEFORE numpy/cv2. cv2 bundles its own
# OpenMP/BLAS runtime DLLs; if they load first, torch's c10.dll initialization
# can collide with them. torch-first is the order the installer's verify step
# uses (and it's known good). This also front-loads the expensive torch import
# to worker startup instead of the first request.
_log(f"worker starting (python {sys.version.split()[0]}, pid {os.getpid()})")
_log("importing torch ...")
import torch  # noqa: E402, F401

_log(f"torch {torch.__version__} (cuda={torch.cuda.is_available()})")

import numpy as np  # noqa: E402


# ── Model state (persists across requests — that's the point of the
#    long-lived worker: SAM stays warm on the GPU between clicks) ──
_predictor = None
_sam3_model = None
_model_key = None


def _free_models():
    global _predictor, _sam3_model, _model_key
    _predictor = None
    _sam3_model = None
    _model_key = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from H2_SamViT_Gizmo import matanyone2_refiner

        matanyone2_refiner.unload()
    except Exception:
        pass


def _load_sam(model):
    """Load (or reuse) the SAM predictor described by the request's model dict."""
    global _predictor, _sam3_model, _model_key
    from H2_SamViT_Gizmo import model_manager

    family = model["family"]
    version = model["version"]
    size = model["size"]
    precision = model["precision"]
    key = (family, version, size, precision)

    if _predictor is not None and _model_key == key:
        return _predictor

    _free_models()

    if not model_manager.is_downloaded(family, version, size):
        if not model.get("allow_download"):
            raise RuntimeError(
                "checkpoint-missing: run Download Model or confirm the download."
            )
        model_manager.download(family, version, size)

    if family == "SAM3":
        _sam3_model = model_manager.build_sam3_model(precision=precision)
        _predictor = _sam3_model.inst_interactive_predictor
    else:
        _predictor = model_manager.build_sam2_predictor(
            version=version, size=size, precision=precision
        )
    _model_key = key
    return _predictor


def _read_image(path):
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read input frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_mask(path, mask):
    """Write the final mask as 16-bit grayscale PNG (preserves soft alpha)."""
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mask16 = (np.clip(mask, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if not cv2.imwrite(path, mask16):
        raise RuntimeError(f"Could not write mask: {path}")


def _predict_point_bbox(req, image):
    """Port of the prompt math from inference.run_point_bbox_inference.

    Coordinates arrive in Nuke space (bottom-left origin) and are flipped to
    SAM space (top-left origin) here, using the image height.
    """
    import torch
    from H2_SamViT_Gizmo import inference

    predictor = _load_sam(req["model"])
    img_h = image.shape[0]
    points = req.get("points") or []
    bbox = req.get("bbox")
    neg_bbox = req.get("neg_bbox")

    amp_ctx = inference._resolve_autocast_ctx(req["model"]["precision"])

    with torch.inference_mode(), amp_ctx:
        predictor.set_image(image)

        point_coords = None
        point_labels = None
        box = None

        if points:
            point_coords = np.array(
                [[p["x"], img_h - p["y"]] for p in points], dtype=np.float32
            )
            point_labels = np.array([p["label"] for p in points], dtype=np.int32)

        if bbox:
            x1, y1_nk, x2, y2_nk = bbox
            box = np.array(
                [x1, img_h - y2_nk, x2, img_h - y1_nk], dtype=np.float32
            )

        # Negative bbox -> its 4 corners + center as background points.
        if neg_bbox:
            nx1, ny1_nk, nx2, ny2_nk = neg_bbox
            neg_corners = np.array(
                [
                    [nx1, img_h - ny2_nk],
                    [nx2, img_h - ny2_nk],
                    [nx1, img_h - ny1_nk],
                    [nx2, img_h - ny1_nk],
                    [(nx1 + nx2) / 2, (img_h - ny2_nk + img_h - ny1_nk) / 2],
                ],
                dtype=np.float32,
            )
            neg_labels = np.zeros(len(neg_corners), dtype=np.int32)
            if point_coords is not None:
                point_coords = np.concatenate([point_coords, neg_corners], axis=0)
                point_labels = np.concatenate([point_labels, neg_labels], axis=0)
            else:
                point_coords = neg_corners
                point_labels = neg_labels

        # 1 ambiguous prompt -> multimask; clear intent -> single mask.
        num_prompts = len(points) + (1 if bbox else 0)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=num_prompts <= 1,
        )

    best = int(np.argmax(scores))
    return masks[best], f"Inference complete. Score: {float(scores[best]):.3f}"


def _predict_text(req, image):
    """Port of the text paths: SAM3's Sam3Processor, or Grounding DINO + SAM2."""
    import torch
    from H2_SamViT_Gizmo import inference

    family = req["model"]["family"]
    text = req["text"]
    sel_points = req.get("selection_points") or []
    amp_ctx = inference._resolve_autocast_ctx(req["model"]["precision"])

    if family == "SAM3":
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor

        _load_sam(req["model"])
        processor = Sam3Processor(_sam3_model)

        with torch.inference_mode(), amp_ctx:
            state = processor.set_image(Image.fromarray(image))
            output = processor.set_text_prompt(text, state)

        masks, scores = output["masks"], output["scores"]
        boxes = output.get("boxes", None)
        if hasattr(masks, "cpu"):
            masks = masks.cpu().float().numpy()
        if hasattr(scores, "cpu"):
            scores = scores.cpu().float().numpy()
        if boxes is not None and hasattr(boxes, "cpu"):
            boxes = boxes.cpu().float().numpy()

        if len(masks) == 0:
            raise RuntimeError(f"no-detections: No objects matching '{text}' found.")

        if sel_points and boxes is not None and len(boxes) > 0:
            idx = inference.select_detection_by_point(boxes, sel_points[0])
        else:
            idx = int(np.argmax(scores))

        coarse = masks[idx].astype(np.float32)
        while coarse.ndim > 2:
            coarse = coarse.squeeze(0)

        # Optional: refine with the detected box as a SAM prompt.
        if boxes is not None and len(boxes) > 0:
            try:
                predictor = _sam3_model.inst_interactive_predictor
                sam_box = np.asarray(boxes[idx][:4])
                with torch.inference_mode(), amp_ctx:
                    predictor.set_image(image)
                    r_masks, r_scores, _ = predictor.predict(
                        box=sam_box, multimask_output=False
                    )
                refined = r_masks[int(np.argmax(r_scores))].astype(np.float32)
                while refined.ndim > 2:
                    refined = refined.squeeze(0)
                if refined.sum() > coarse.sum() * 0.5:
                    coarse = np.maximum(coarse, refined)
                    _log("Text mask refined with box prompt")
            except Exception as e:
                _log(f"Box refinement skipped: {e}")

        return coarse, (
            f"Text inference (SAM3) complete. "
            f"Found {len(masks)} objects, selected #{idx + 1}"
        )

    # ── SAM2 path: Grounding DINO detection -> SAM2 segmentation ──
    from H2_SamViT_Gizmo import inference as inf

    text_detector = inf.load_text_model()
    predictor = _load_sam(req["model"])

    model = text_detector["model"]
    processor = text_detector["processor"]
    device = text_detector["device"]

    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=0.2,
        text_threshold=0.2,
        target_sizes=[(image.shape[0], image.shape[1])],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    if len(boxes) == 0:
        raise RuntimeError(f"no-detections: No objects matching '{text}' found.")

    if sel_points:
        idx = inference.select_detection_by_point(boxes, sel_points[0])
    else:
        idx = int(np.argmax(scores))

    with torch.inference_mode(), amp_ctx:
        predictor.set_image(image)
        masks, m_scores, _ = predictor.predict(
            box=boxes[idx], multimask_output=False
        )

    return masks[int(np.argmax(m_scores))], (
        f"Text inference (SAM2) complete. "
        f"Found {len(boxes)} objects, selected #{idx + 1}"
    )


def _handle_infer(req):
    from H2_SamViT_Gizmo import inference

    image = _read_image(req["image"])

    if req["mode"] == "text":
        coarse, message = _predict_text(req, image)
    else:
        coarse, message = _predict_point_bbox(req, image)

    mask, soft = inference.finalize_mask(
        image, coarse, req.get("params") or {}, bool(req.get("use_ma2"))
    )
    _write_mask(req["out_mask"], mask)
    return {"ok": True, "message": message, "soft": soft}


def _handle(req):
    cmd = req.get("cmd")
    if cmd == "ping":
        import torch

        return {
            "ok": True,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        }
    if cmd == "infer":
        return _handle_infer(req)
    if cmd == "clear":
        _free_models()
        return {"ok": True}
    if cmd == "quit":
        return None  # sentinel: reply then exit
    raise RuntimeError(f"unknown command: {cmd!r}")


def _process(line, send) -> bool:
    """Handle one raw request line. Returns True when the worker should quit."""
    line = line.strip()
    if not line:
        return False
    try:
        req = json.loads(line)
    except Exception:
        send({"ok": False, "error": "bad request (invalid JSON)"})
        return False
    try:
        resp = _handle(req)
    except Exception as e:
        _log(traceback.format_exc())
        send({"ok": False, "error": str(e)})
        return False
    if resp is None:  # quit
        send({"ok": True})
        return True
    send(resp)
    return False


def _stdio_main():
    def send(obj):
        _PROTO.write(json.dumps(obj) + "\n")
        _PROTO.flush()

    for line in sys.stdin:
        if _process(line, send):
            break
    _log("worker exiting")


def _sock_readline(conn, buf):
    """Read one \\n-terminated line from conn (buffered in buf[0]).

    Returns the line without the newline, b"" on peer close, or None on a
    read-timeout tick (so the caller can check the idle budget).
    """
    import socket

    while b"\n" not in buf[0]:
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            return None
        except OSError:
            return b""
        if not chunk:
            return b""
        buf[0] += chunk
    line, _, rest = buf[0].partition(b"\n")
    buf[0] = rest
    return line


def _serve_main(rdv):
    """Detached socket mode: advertise {port, token, pid} in the rendezvous
    file, then serve one client at a time until quit or idle timeout."""
    import secrets
    import socket

    IDLE_LIMIT = 1800.0  # self-destruct after 30 min without a request

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(15.0)
    port = srv.getsockname()[1]
    token = secrets.token_hex(16)

    tmp = rdv + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"port": port, "token": token, "pid": os.getpid()}, f)
    os.replace(tmp, rdv)
    _log(f"serving on 127.0.0.1:{port}")

    last = time.time()
    running = True
    try:
        while running:
            if time.time() - last > IDLE_LIMIT:
                _log("idle timeout — exiting")
                break
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.settimeout(20.0)
            buf = [b""]

            def send(obj, _c=conn):
                _c.sendall((json.dumps(obj) + "\n").encode("utf-8"))

            try:
                hello_line = _sock_readline(conn, buf)
                hello = json.loads(hello_line) if hello_line else {}
                if hello.get("token") != token:
                    _log("bad/missing token — closing connection")
                    conn.close()
                    continue
                send({"ok": True, "hello": True})
                _log("client connected")
                last = time.time()
                while True:
                    line = _sock_readline(conn, buf)
                    if line is None:  # timeout tick — check idle budget
                        if time.time() - last > IDLE_LIMIT:
                            _log("idle timeout — exiting")
                            running = False
                            break
                        continue
                    if line == b"":  # peer closed (Nuke exit/reload)
                        break
                    last = time.time()
                    if _process(line.decode("utf-8", "replace"), send):
                        running = False
                        break
            except Exception:
                _log(traceback.format_exc())
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                _log("client disconnected")
    finally:
        try:
            os.remove(rdv)
        except OSError:
            pass
    _log("worker exiting")


def main():
    if _RDV:
        _serve_main(_RDV)
    else:
        _stdio_main()


if __name__ == "__main__":
    main()
