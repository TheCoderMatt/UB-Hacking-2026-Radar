"""
Spacecraft Debris Radar - Acconeer XE125 + webcam  (standalone - no other files needed)
=======================================================================================

The XE125 is the nose of a spaceship pointing along its heading. Objects in
front of it are asteroids. For each one the program shows its position, how
fast it is closing in, and its TIME TO IMPACT:

    time to impact = distance / closing speed

It also checks whether the asteroid will actually cross the ship's path (a
corridor as wide as the ship, straight ahead) and raises an alert:
    BRACE      within 5 cm: BRACE FOR IMPACT        (flashing red/white)
    COLLISION  within 10 cm (any direction), or in the ship's path with
               impact in under 3 s                         (red)
    WARNING    in the ship's path, impact in under 10 s or
               within 20 cm                                (amber)
    IN PATH    in the ship's path but not approaching      (yellow)
    CLEAR      will pass to the side                       (green)
It also PLANS A ROUTE through the asteroids (drawn in light blue): it keeps
straight if it can, otherwise it goes around, or SNAKES left and right through
gaps when there's room, then comes back onto its heading. The banner shows the
manoeuvre (HOLD COURSE / STEER LEFT / GO AROUND RIGHT / SNAKE L>R) and the line
beside the ship shows where each turn starts. If there's no gap wide enough
for the ship, it says NO GAP - REVERSE THRUST.

How it works underneath:

The XE125 measures DISTANCE but not direction. Your laptop's webcam sees
DIRECTION (left-right) but not distance. This program combines the two, so
each object gets a real 2D position on the radar screen.

No markers needed: when you press  b , both the radar and the camera record
the EMPTY scene. Anything that appears afterwards is treated as an object.

Setup:
  * The laptop's front camera faces the person at the screen, so the radar
    must face the SAME way: put the radar just in front of the screen,
    centred below the camera, pointing at the table in front of the laptop.
  * Only the "table zone" of the camera image is watched (amber outline;
    everything outside it is shaded), so you moving around behind the
    objects is ignored. DRAW the zone around the table's edge in the camera
    view to change it (see "Mouse" below).
  * Keep lighting steady; big lighting changes look like objects. Press  b
    again whenever the empty scene changes.

Routine:
  1. Clear the table and press  b  to record the EMPTY scene.
  2. Place your objects. Auto-tune adjusts the settings by itself every few
     seconds (it waits while anything is moving).

Buttons (bottom right):
  AUTO-TUNE ON/OFF continuous auto-tune: every 3 s it looks at the last 2 s
                   and sets the radar threshold, radar focus and camera
                   sensitivity so every steady object is detected while
                   flickering noise is ignored. Any manual adjustment
                   pauses it; click (or press  a ) to resume.
  RESET SETTINGS   default settings, default table zone, auto-tune back on

Keys (click the window first so it has keyboard focus):
  b          record the EMPTY scene (table clear, hands away)
  a          turn auto-tune on / off
  r          same as RESET SETTINGS
  m          mirror left-right (if dots move the wrong way)
  up / down  raise / lower the RADAR detection threshold
  [ / ]      lower / raise the CAMERA sensitivity threshold
  , / .      loosen / tighten the radar FOCUS (keep weaker / only stronger echoes)
  q          quit
Mouse, in the camera view - drawing the table zone:
  click, click, click...     place corner points around the table
  hold and trace             draw the outline freehand; release to finish
  double-click or Enter      finish a click-by-click outline
  click the first point      also finishes it
  right-click                remove the last point
  Esc                        cancel (keeps the previous zone)

Focus: the program keeps every radar echo that is reasonably strong compared
with the STRONGEST echo right now (at least 25% of it by default), and drops
weak ones such as hands, noise, and faint reflections. Camera shapes touching
the zone's outline and continuing outside it (usually arms reaching in) or
filling most of the zone (a person or a lighting change) are ignored.

Usage (inside radar-env, Exploration Tool disconnected):
  python radar_fusion.py --port COM3
  python radar_fusion.py --port COM3 --focus 0.4          # stricter: echoes >= 40% of strongest
  python radar_fusion.py --port COM3 --objects 2          # also cap at the 2 strongest
  python radar_fusion.py --port COM3 --max-distance 50    # ignore radar echoes beyond 50 cm
  python radar_fusion.py --port COM3 --ship-width 15      # wider ship (cm)
  python radar_fusion.py --port COM3 --heading 90         # compass heading shown on screen
  python radar_fusion.py --port COM3 --warn 8 --danger 2  # alert timings (seconds)
  python radar_fusion.py --port COM3 --danger-distance 8 --warn-distance 15   # cm
  python radar_fusion.py --port COM3 --boost 3   # 3x radar averaging: weak objects stand out
  python radar_fusion.py --port COM3 --brace-distance 4  # BRACE FOR IMPACT distance (cm)
  python radar_fusion.py --port COM3 --near-band 0.4   # widen "close to radar" camera band
  python radar_fusion.py --port COM3 --hfov 75   # camera's horizontal field of view
  python radar_fusion.py --port COM3 --camera 1  # pick a different camera
  python radar_fusion.py --port COM3 --far       # radar range 10 cm to 1.5 m

Requires OpenCV:  python -m pip install opencv-python
"""

import argparse
import collections
import queue
import sys
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from acconeer.exptool import a121
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button

POINT_M = 2.5e-3              # one A121 distance "point" is 2.5 mm
GREEN = "#39ff14"
DIM_GREEN = "#1f7a12"
AMBER = "#ffb000"
TRACK_COLORS = ["#39ff14", "#00e5ff", "#ffb000", "#ff4fd8", "#ffffff"]
BEAM_HALF_ANGLE_DEG = 30      # drawn radar beam width (visual only)
DISPLAY_FPS = 10              # screen redraws per second
MAX_TARGETS = 5               # most radar objects detected per frame
MAX_BLOBS = 5                 # most camera objects detected per frame
CAM_W, CAM_H = 320, 240       # camera frames are shrunk to this for speed
CAMERA_MEMORY_S = 3.0         # keep a lost camera direction this long
BACKGROUND_FRAMES = 30        # radar frames in a background recording (~3 s)
TUNE_FRAMES = 20              # radar frames each auto-tune looks at (~2 s)
TUNE_EVERY_S = 2.0            # how often the continuous auto-tune runs
TUNE_CAMERA_FRAMES = 3        # camera frames each auto-tune looks at
DEFAULTS = {"radar": 5.0, "camera": 30.0, "focus": 0.25}

# Near the sensor the radar's own transmit signal leaks into its receiver (plus
# a reflection off the housing). That leakage drifts as the chip warms up, and
# the leftover can look like a fake object a few cm away. So close in:
NEAR_ZONE_END_M = 0.10        # this close to the sensor counts as the "near zone"
NEAR_BOOST = 3.0              # threshold is this many times higher at the sensor,
                              # easing back to normal at the end of the near zone
NEAR_DRIFT_RATE = 0.05        # near-zone background follows slow drift (~2 s)
NEAR_DRIFT_LIMIT = 3.0        # ...but only changes smaller than this x the noise, so
                              # a real object placed close is never absorbed
CLOSE_PAIR_M = 0.15           # a radar echo this close only pairs with a camera
                              # object near the bottom (radar end) of the table zone

# Spaceship view
RED = "#ff3b3b"
YELLOW = "#ffe14d"
WHITE = "#ffffff"
THREAT_COLORS = {"BRACE": RED, "COLLISION": RED, "WARNING": AMBER, "IN PATH": YELLOW,
                 "CLEAR": GREEN}
THREAT_RANK = {"BRACE": -1, "COLLISION": 0, "WARNING": 1, "IN PATH": 2, "CLEAR": 3}
MIN_CLOSING = 0.02            # m/s: slower than 2 cm/s counts as not approaching
PATH_MARGIN = 0.02            # m: extra clearance around the ship's width
ASTEROID_SHAPE = [(1.0, 0.1), (0.75, 0.7), (0.15, 1.0), (-0.55, 0.85), (-1.0, 0.25),
                  (-0.85, -0.55), (-0.25, -1.0), (0.5, -0.85), (1.0, 0.1)]


# =============================================================== radar ====

def make_config(far, boost=1):
    """Radar settings. `boost` multiplies HWAAS (hardware averaging).

    The XE125's transmit power is fixed, but averaging more pulses per
    measurement on the chip lowers the noise, so weak echoes stand out more:
    3x the averaging gives about 3x less noise power (~+5 dB). It is done on
    the radar itself, so no extra data goes over USB.
    """
    if far:
        start_m, end_m, step_length, profile, hwaas = 0.10, 1.50, 2, a121.Profile.PROFILE_3, 16
    else:
        start_m, end_m, step_length, profile, hwaas = 0.03, 0.60, 1, a121.Profile.PROFILE_1, 32
    hwaas = int(min(hwaas * boost, 511))
    start_point = int(round(start_m / POINT_M))
    num_points = int((end_m - start_m) / POINT_M) // step_length
    return a121.SensorConfig(start_point=start_point, step_length=step_length,
                             num_points=num_points, profile=profile, hwaas=hwaas,
                             sweeps_per_frame=4, frame_rate=10.0)


def distances_m(config):
    return (config.start_point + np.arange(config.num_points) * config.step_length) * POINT_M


class RadarDetector:
    """Background subtraction + threshold detection of every echo peak."""

    def __init__(self, dist, min_separation_m):
        self.dist = dist
        self.min_separation_m = min_separation_m
        self.background = None
        self.noise = None
        self.factor = 5.0
        self.history = collections.deque(maxlen=2)
        # Threshold multiplier per distance: NEAR_BOOST at the sensor, 1 beyond the near zone
        span = max(NEAR_ZONE_END_M - dist[0], 1e-6)
        closeness = np.clip((NEAR_ZONE_END_M - dist) / span, 0.0, 1.0)
        self.near_boost = 1.0 + (NEAR_BOOST - 1.0) * closeness
        self.near = dist < NEAR_ZONE_END_M

    @staticmethod
    def frame_vector(result):
        return result.frame.mean(axis=0)

    def set_background(self, vectors):
        vectors = np.asarray(vectors)
        self.background = vectors.mean(axis=0).astype(complex)
        leftover = np.abs(vectors - self.background)
        self.noise = np.maximum(np.percentile(leftover, 90, axis=0), np.median(leftover)) + 1e-9
        self.history.clear()

    def update(self, vec):
        if self.background is None:
            amp = np.abs(vec)
            threshold = np.full_like(amp, np.inf)
        else:
            amp = np.abs(vec - self.background)
            threshold = self.factor * self.noise * self.near_boost
            # Near the sensor, let the background follow slow leakage drift
            # while it is still small, so it never builds up into a fake
            # object. A real object is a big, sudden change and is left alone.
            quiet = self.near & (amp < NEAR_DRIFT_LIMIT * self.noise * self.near_boost)
            if quiet.any():
                self.background[quiet] += NEAR_DRIFT_RATE * (vec[quiet] - self.background[quiet])
        self.history.append(amp)
        amp = np.mean(self.history, axis=0)
        return amp, threshold, self._targets(amp, threshold)

    def _targets(self, amp, threshold):
        n = len(amp)
        inner = np.arange(1, n - 1)
        is_peak = (amp[inner] >= amp[inner - 1]) & (amp[inner] > amp[inner + 1])
        candidates = inner[is_peak & (amp[inner] > threshold[inner])]
        if amp[0] > threshold[0] and amp[0] > amp[1]:
            candidates = np.r_[0, candidates]
        step = self.dist[1] - self.dist[0]
        min_gap = max(1, int(round(self.min_separation_m / step)))
        kept = []
        for k in sorted(candidates, key=lambda k: amp[k], reverse=True):
            if all(abs(k - other) >= min_gap for other in kept):
                kept.append(int(k))
            if len(kept) == MAX_TARGETS:
                break
        targets = []
        for k in sorted(kept):
            offset = 0.0
            if 0 < k < n - 1:
                a, b, c = amp[k - 1], amp[k], amp[k + 1]
                denom = a - 2 * b + c
                if denom != 0:
                    offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
            targets.append((float(self.dist[k] + offset * step), float(amp[k])))
        return targets


class Track:
    def __init__(self, track_id, distance, strength, t):
        self.id = track_id
        self.distance = distance
        self.strength = strength
        self.velocity = 0.0
        self.last_time = t
        self.hits = 1
        self.misses = 0
        self.recent = collections.deque([True], maxlen=10)  # seen / not seen, last 10 frames
        self.suspect = False      # near-sensor echo the camera can't confirm

    @property
    def confirmed(self):
        # Seen in at least 3 frames AND in most recent frames: a steady object,
        # not an echo that flickers on and off.
        return self.hits >= 3 and sum(self.recent) >= 0.8 * len(self.recent)

    @property
    def coasting(self):
        return self.misses > 0

    def predict(self, t):
        dt = max(t - self.last_time, 1e-3)
        return self.distance + self.velocity * dt, dt


class Tracker:
    """Keeps a persistent ID per radar object and follows it frame to frame."""

    GATE_M, ALPHA, BETA, MAX_MISSES = 0.06, 0.6, 0.25, 6

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections, t):
        predictions = {tr.id: tr.predict(t) for tr in self.tracks}
        pairs = []
        for tr in self.tracks:
            pred, _ = predictions[tr.id]
            for j, (d, _) in enumerate(detections):
                if abs(d - pred) <= self.GATE_M:
                    pairs.append((abs(d - pred), tr, j))
        pairs.sort(key=lambda p: p[0])
        used_tracks, used_dets = set(), set()
        for _, tr, j in pairs:
            if tr.id in used_tracks or j in used_dets:
                continue
            used_tracks.add(tr.id)
            used_dets.add(j)
            pred, dt = predictions[tr.id]
            residual = detections[j][0] - pred
            tr.distance = pred + self.ALPHA * residual
            tr.strength = 0.7 * tr.strength + 0.3 * detections[j][1]
            tr.velocity += self.BETA * residual / dt
            tr.last_time = t
            tr.hits += 1
            tr.misses = 0
            tr.recent.append(True)
        for tr in self.tracks:
            if tr.id not in used_tracks:
                tr.misses += 1
                tr.recent.append(False)
                tr.distance = predictions[tr.id][0]
                tr.velocity *= 0.8
                tr.last_time = t
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.MAX_MISSES]
        for j, (d, strength) in enumerate(detections):
            if j not in used_dets:
                self.tracks.append(Track(self._free_id(), d, strength, t))
        return [tr for tr in self.tracks if tr.confirmed]

    def _free_id(self):
        """Smallest ID not in use, so labels stay short (T1, T2...)."""
        in_use = {tr.id for tr in self.tracks}
        i = 1
        while i in in_use:
            i += 1
        return i


def direction_word(velocity):
    if velocity < -0.02:
        return "APPROACHING"
    if velocity > 0.02:
        return "MOVING AWAY"
    return "STILL"


# ============================================================== camera ====

class Camera:
    """Reads the webcam in a background thread and keeps the latest frame."""

    def __init__(self, index):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. Close other apps using the "
                               f"camera, or try --camera 1")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


class CameraDetector:
    # Default table zone: the lower 60% of the picture
    DEFAULT_ZONE = ((0, int(CAM_H * 0.4)), (CAM_W - 1, int(CAM_H * 0.4)),
                    (CAM_W - 1, CAM_H - 1), (0, CAM_H - 1))

    """Finds objects by comparing each frame with a picture of the empty scene.

    To keep boxes tight around each object it:
      * removes SHADOWS (pixels that got darker but kept their colour),
      * boxes only the object's strongly-changed core, not its faint fringe,
      * splits a shape into separate objects where it narrows into a "waist"
        (two objects side by side that touch in the picture).
    """

    def __init__(self):
        self.background = None        # blurred empty scene (float)
        self.background_hsv = None
        self.collected = []
        self.threshold = 30.0                           # change that counts as "object"
        # The table zone: an outline drawn by the user (default: lower 60%)
        self.zone_points = list(self.DEFAULT_ZONE)
        self.zone = self._fill(self.zone_points)
        self.max_objects = MAX_BLOBS
        self.last_mask = None

    @staticmethod
    def _fill(points):
        mask = np.zeros((CAM_H, CAM_W), np.uint8)
        cv2.fillPoly(mask, [np.round(np.array(points)).astype(np.int32)], 1)
        return mask.astype(bool)

    def set_zone(self, points):
        """Use a new outline. Returns False if it's too small to be useful."""
        zone = self._fill(points)
        if zone.sum() < 400:
            return False
        self.zone_points = [tuple(p) for p in points]
        self.zone = zone
        return True

    def reset_zone(self):
        self.zone_points = list(self.DEFAULT_ZONE)
        self.zone = self._fill(self.zone_points)

    @staticmethod
    def _blur(frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def start_background(self):
        self.collected = []

    def add_background(self, frame):
        self.collected.append(self._blur(frame).astype(np.float32))

    def finish_background(self):
        if self.collected:
            self.background = np.median(np.stack(self.collected), axis=0)
            self.background_hsv = cv2.cvtColor(self.background.astype(np.uint8),
                                               cv2.COLOR_BGR2HSV).astype(np.float32)
        self.collected = []

    def _shadow_mask(self, blurred):
        """Pixels that look like shadow: darker, but same colour as before."""
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg = self.background_hsv
        ratio = (hsv[..., 2] + 1) / (bg[..., 2] + 1)
        dh = np.abs(hsv[..., 0] - bg[..., 0])
        dh = np.minimum(dh, 180 - dh)
        ds = np.abs(hsv[..., 1] - bg[..., 1])
        colourful = bg[..., 1] > 40
        same_colour = (ds < 40) & (~colourful | (dh < 12))
        return (ratio > 0.35) & (ratio < 0.93) & same_colour

    def find(self, frame, threshold=None):
        """Objects in the table zone: list of dicts with centre_x, bottom_y, box, area.
        `threshold` overrides the current sensitivity (used by auto-tune)."""
        if self.background is None:
            return []
        thr = self.threshold if threshold is None else threshold
        blurred = self._blur(frame)
        diff = np.abs(blurred.astype(np.float32) - self.background).max(axis=2)
        diff -= np.median(diff)                         # cancel overall brightness shifts
        fg = diff > thr
        shadow = self._shadow_mask(blurred)

        zone = self.zone
        # Find changed shapes in the WHOLE picture first, so we can tell whether
        # a shape lies inside the zone or reaches in from outside it.
        full = fg.astype(np.uint8) * 255
        full = cv2.morphologyEx(full, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.last_mask = full
        core = (diff > 1.5 * thr) & (full > 0) & ~shadow  # strongly changed

        zone_area = int(zone.sum())
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full)
        blobs = []
        for lab in range(1, n_labels):
            if stats[lab][4] < 120:                     # too small: noise
                continue
            whole = labels == lab
            region = whole & zone
            inside = int(region.sum())
            if inside < 120:                            # (almost) entirely outside the zone
                continue
            if (whole & ~zone).sum() > 0.15 * inside:   # continues outside: an arm reaching in
                continue
            if inside > 0.4 * zone_area:                # fills the zone: a person or lighting
                continue
            # Trim shadow-like pixels only when a clear object is attached to
            # them. A shape that is ALL shadow-like is kept whole: it may be a
            # grey object that is just a little darker than the table.
            solid = region & ~shadow
            if solid.sum() >= 0.3 * region.sum():
                solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_OPEN,
                                         np.ones((3, 3), np.uint8)) > 0
                if solid.sum() >= 60:
                    region = solid
                    region_core = core
                else:
                    region_core = region
            else:
                region_core = region
            for part in self._split(region, region_core & region):
                blobs.append(part)

        blobs.sort(key=lambda b: b["area"], reverse=True)
        return blobs[:self.max_objects]

    @staticmethod
    def _split(region, core):
        """Split one shape into separate objects, then box each piece's core tightly.

        Splits happen at narrow "waists" (two objects side by side that barely
        touch) and where the shape's top or bottom edge jumps sharply (a near
        object overlapping one behind it: their bases sit at different heights
        in the picture).
        """
        cols = region.sum(axis=0)
        xs = np.flatnonzero(cols)
        peak = cols.max()
        on = cols >= max(3, 0.25 * peak)                # columns that are solidly object

        # Top and bottom edge of the shape in each column (smoothed a little)
        height = region.shape[0]
        rows = np.arange(height)[:, None]
        top = np.where(region.any(axis=0), np.where(region, rows, height).min(axis=0), -1)
        bottom = np.where(region.any(axis=0), np.where(region, rows, -1).max(axis=0), -1)
        top_s = np.round(cv2.medianBlur(top.astype(np.float32).reshape(1, -1), 5)).ravel()
        bot_s = np.round(cv2.medianBlur(bottom.astype(np.float32).reshape(1, -1), 5)).ravel()
        blob_h = bottom[xs].max() - top[xs][top[xs] >= 0].min() + 1
        jump = max(10, 0.2 * blob_h)

        pieces, start = [], None
        for x in range(xs[0], xs[-1] + 2):
            inside = x <= xs[-1] and on[x]
            edge_jump = (inside and start is not None and x - start >= 6 and
                         (abs(bot_s[x] - bot_s[x - 1]) > jump or abs(top_s[x] - top_s[x - 1]) > jump))
            if edge_jump:                               # end this piece, start a new one here
                pieces.append((start, x))
                start = x
            elif inside:
                start = x if start is None else start
            elif start is not None:
                if x - start >= 6:                      # at least 6 px wide
                    pieces.append((start, x))
                start = None
        pieces = [(a, b) for a, b in pieces if b - a >= 6]
        if not pieces:
            pieces = [(xs[0], xs[-1] + 1)]

        parts = []
        for a, b in pieces:
            piece = np.zeros_like(region)
            piece[:, a:b] = region[:, a:b]
            strong = piece & core
            use = strong if strong.sum() >= 30 else piece
            # Keep only rows/columns that are solidly part of the object, which
            # trims stray fringe pixels and slivers of a neighbouring object.
            row_count, col_count = use.sum(axis=1), use.sum(axis=0)
            good_rows = np.flatnonzero(row_count >= max(2, 0.3 * row_count.max()))
            good_cols = np.flatnonzero(col_count >= max(2, 0.3 * col_count.max()))
            bx0, bx1 = int(good_cols[0]), int(good_cols[-1])
            by0, by1 = int(good_rows[0]), int(good_rows[-1])
            bw, bh = max(bx1 - bx0 + 1, 2), max(by1 - by0 + 1, 2)
            parts.append({"centre_x": bx0 + bw / 2, "bottom_y": by0 + bh,
                          "box": (bx0, by0, bw, bh), "area": int(piece.sum())})
        return parts


def pixel_to_angle_deg(cx, hfov_deg, mirror):
    """Horizontal pixel position -> direction in degrees (positive = right)."""
    focal_px = (CAM_W / 2) / np.tan(np.radians(hfov_deg / 2))
    angle = np.degrees(np.arctan((cx - CAM_W / 2) / focal_px))
    return -angle if mirror else angle


# ============================================================== fusion ====

def near_band_limit(zone_points, near_band):
    """Image row below which a camera object counts as close to the radar.

    The bottom of the table zone is the end nearest the radar, so 'close'
    means in the lowest `near_band` fraction of the zone's height.
    """
    ys = [p[1] for p in zone_points]
    top, bottom = min(ys), max(ys)
    return bottom - near_band * (bottom - top)


def fuse(tracks, blobs, memory, now, near_row=None, camera_ok=False):
    """Pair radar tracks (distance) with camera blobs (direction).

    On a table, a farther object sits higher in the camera image, so camera
    blobs are ordered lowest-first and paired with radar tracks ordered
    nearest-first.

    Distance check: a radar echo closer than CLOSE_PAIR_M is only paired with
    a camera object near the bottom of the table zone (image row >= near_row).
    An object sitting halfway up the table can't be 6 cm from the radar, so
    that pair would be wrong: the camera object stays "direction only" and the
    echo stays radar-only. If the camera is working and sees nothing near the
    radar, a close radar-only echo is marked suspect (likely a ghost).

    Returns (fused, radar_only, camera_only):
      fused       - (track, blob or None, angle_deg, from_memory)
      radar_only  - tracks with no direction
      camera_only - blobs with no radar echo
    """
    blobs = sorted(blobs, key=lambda b: b["bottom_y"], reverse=True)
    for blob in blobs:
        blob["near"] = near_row is None or blob["bottom_y"] >= near_row
    any_near_blob = any(b["near"] for b in blobs)
    for tr in tracks:
        tr.suspect = False

    def is_close(tr):
        return near_row is not None and tr.distance < CLOSE_PAIR_M

    # Close echoes with no close camera object can't pair with anything
    can_pair = [tr for tr in tracks if not is_close(tr) or any_near_blob]
    cannot = [tr for tr in tracks if tr not in can_pair]
    if len(can_pair) > len(blobs):
        # More radar echoes than camera objects: the strongest echoes are most
        # likely the objects the camera sees; the rest stay radar-only.
        strongest = sorted(can_pair, key=lambda tr: tr.strength, reverse=True)[:len(blobs)]
        rest = [tr for tr in can_pair if tr not in strongest]
    else:
        strongest, rest = can_pair, []
    ordered = sorted(strongest, key=lambda tr: tr.distance)

    fused, radar_only, unpaired = [], [], list(cannot) + rest
    free = list(blobs)
    for tr in ordered:
        if not free:
            unpaired.append(tr)
            continue
        blob = free[0]
        if is_close(tr) and not blob["near"]:
            unpaired.append(tr)                 # distances disagree: don't pair
            continue
        free.pop(0)
        fused.append((tr, blob, blob["angle"], False))
        memory[tr.id] = (blob["angle"], now)
    for tr in sorted(unpaired, key=lambda tr: tr.distance):
        remembered = memory.get(tr.id)
        if remembered and now - remembered[1] <= CAMERA_MEMORY_S:
            fused.append((tr, None, remembered[0], True))
        else:
            if camera_ok and is_close(tr) and not any_near_blob:
                tr.suspect = True
            radar_only.append(tr)
    live = {tr.id for tr in tracks}
    for track_id in list(memory):
        if track_id not in live:
            del memory[track_id]
    return fused, radar_only, free


def focus_tracks(tracks, fraction, cap):
    """Keep echoes at least `fraction` as strong as the strongest one, nearest first."""
    if not tracks:
        return []
    # Near-zone echoes (where the leakage is) don't set the reference, so a
    # strong ghost at the sensor can't push real, weaker objects out.
    reference = [tr for tr in tracks if tr.distance >= NEAR_ZONE_END_M] or tracks
    strongest = max(tr.strength for tr in reference)
    keep = [tr for tr in tracks if tr.strength >= fraction * strongest]
    if cap > 0:
        keep = sorted(keep, key=lambda tr: tr.strength, reverse=True)[:cap]
    return sorted(keep, key=lambda tr: tr.distance)


def auto_tune(radar_amps, cam_frames, radar_det, cam_det):
    """Choose settings from the last ~2 s of radar data and a few camera frames.

    Runs automatically every few seconds, in a background thread.

    Radar: every echo peak is collected at a deliberately low bar, then grouped
    by distance. Peaks present in most frames are OBJECTS; peaks that come and
    go are FLICKER (noise). The threshold goes between the two, leaning toward
    keeping the objects, and the focus is set so the weakest object is kept.
    With no objects at all, the threshold is set just above the flicker so a
    new object is picked up straight away.

    Camera: the sensitivity is swept, and the value is chosen where the camera
    finds the same number of objects as the radar (or none, on an empty
    table), from the middle of the range where that count is steady.

    Returns a dict of new settings, or None if the scene is changing too much
    to judge (something moving, appearing or disappearing): then nothing is
    changed.
    """
    noise = radar_det.noise
    dist = radar_det.dist
    step = dist[1] - dist[0]
    n_frames = len(radar_amps)

    # ---- radar: group peaks from all frames by distance
    clusters = []                                   # each: distances, amps, ratios
    for amp in radar_amps:
        # Near the sensor, strengths are measured against the raised near-zone
        # bar, so leakage there can't push the threshold up for everything else.
        for d, a in radar_det._targets(amp, 2.0 * noise * radar_det.near_boost):
            idx = int(np.clip(round((d - dist[0]) / step), 0, len(noise) - 1))
            ratio = a / (noise[idx] * radar_det.near_boost[idx])
            for c in clusters:
                if abs(np.mean(c["d"]) - d) < radar_det.min_separation_m / 2:
                    c["d"].append(d)
                    c["a"].append(a)
                    c["r"].append(ratio)
                    break
            else:
                clusters.append({"d": [d], "a": [a], "r": [ratio]})

    steady = [c for c in clusters if len(c["d"]) >= 0.7 * n_frames]
    flicker = [c for c in clusters if len(c["d"]) < 0.4 * n_frames]
    in_between = [c for c in clusters if c not in steady and c not in flicker]
    # Something strong that is only sometimes there is an object appearing,
    # leaving or moving, not noise: don't tune on that.
    if in_between or any(max(c["r"]) > 12 for c in flicker):
        return None
    worst_flicker = max([max(c["r"]) for c in flicker], default=2.5)

    if steady:
        weakest_object = min(np.percentile(c["r"], 20) for c in steady)
        if weakest_object > 1.2 * worst_flicker:
            factor = np.sqrt(weakest_object * worst_flicker)   # halfway (ratio scale)
        else:
            factor = weakest_object / 1.3                      # favour the objects
        strengths = [np.median(c["a"]) for c in steady]
        focus = float(np.clip(0.6 * min(strengths) / max(strengths), 0.05, 0.9))
    else:                                           # empty scene: just above the noise
        factor = 1.3 * worst_flicker
        focus = DEFAULTS["focus"]
    factor = float(np.clip(factor, 2.5, 40.0))
    n_objects = len(steady)

    # ---- camera: sweep the sensitivity and look for a steady, matching count
    camera_thr = None
    if cam_det.background is not None and cam_frames:
        candidates = list(range(12, 84, 6))
        counts = [int(np.median([len(cam_det.find(f, threshold=float(thr))) for f in cam_frames]))
                  for thr in candidates]
        target = min(n_objects, cam_det.max_objects)
        best_gap = min(abs(c - target) for c in counts)
        good = [i for i, c in enumerate(counts) if abs(c - target) == best_gap]
        runs, run = [], [good[0]]                    # longest run of good values...
        for i in good[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                runs.append(run)
                run = [i]
        runs.append(run)
        longest = max(runs, key=len)
        if target == 0:                              # empty: lowest clean value + margin
            camera_thr = float(candidates[longest[0]] + 6)
        else:                                        # ...and its middle
            camera_thr = float(candidates[longest[len(longest) // 2]])

    return {"radar": factor, "focus": focus, "camera": camera_thr, "objects": n_objects}


class ThreatAssessor:
    """Turns tracked objects into asteroids with a time to impact and a threat level.

    Anything within the danger distance (10 cm by default) is an IMMINENT
    collision whatever its speed or direction, and anything in the ship's path
    within the warning distance (20 cm) is a warning.

    The ship sits at the radar, pointing along the radar's beam (its heading).
    Its path is a corridor as wide as the ship (plus a margin) straight ahead.
      time to impact = distance / closing speed           (radar)
      side position  = distance x sin(bearing)            (radar + camera)
    The side position is followed over time to predict where the asteroid will
    be when it reaches the ship, i.e. whether it will hit the corridor or pass.
    """

    def __init__(self, ship_width_m, warn_s, danger_s, danger_m, warn_m, brace_m=0.05):
        self.brace_m = brace_m                # this close: BRACE FOR IMPACT
        self.half_path = ship_width_m / 2 + PATH_MARGIN
        self.warn_s = warn_s
        self.danger_s = danger_s
        self.danger_m = danger_m              # anything this close: imminent collision
        self.warn_m = warn_m                  # in the ship's path and this close: warning
        self.lateral = {}                       # track id -> (time, x, x velocity)
        self.last_steer = {}                    # track id -> last advice, to avoid flip-flopping

    def reset(self):
        self.lateral.clear()
        self.last_steer.clear()

    def _lateral_velocity(self, track_id, x, now):
        if track_id not in self.lateral:
            self.lateral[track_id] = (now, x, 0.0)
            return 0.0
        t0, x0, vx = self.lateral[track_id]
        dt = now - t0
        if dt > 1e-3:
            vx = float(np.clip(0.8 * vx + 0.2 * (x - x0) / dt, -0.3, 0.3))
            self.lateral[track_id] = (now, x, vx)
        return vx

    def _assess(self, tr, angle, x, y, vx, blob, from_memory):
        closing = -tr.velocity                  # m/s, positive = coming closer
        tti = tr.distance / closing if closing > MIN_CLOSING else None
        if x is None:                           # bearing unknown: assume the worst
            x_impact, in_path = None, True
        else:
            ahead = min(tti, self.warn_s) if tti is not None else 0.0
            x_impact = x + vx * ahead
            in_path = abs(x_impact) <= self.half_path
        close = tr.distance <= self.danger_m
        if tr.suspect:
            # Near-sensor echo the camera can't see: probably leakage, not an
            # object. Still flagged, but not as a red collision.
            level = "WARNING" if in_path else "CLEAR"
        elif tr.distance <= self.brace_m:
            level = "BRACE"
        elif close or (in_path and tti is not None and tti <= self.danger_s):
            level = "COLLISION"
        elif in_path and ((tti is not None and tti <= self.warn_s) or
                          tr.distance <= self.warn_m):
            level = "WARNING"
        elif in_path:
            level = "IN PATH"
        else:
            level = "CLEAR"
        if x_impact is None:
            steer = None
        else:
            steer = "LEFT" if x_impact >= 0 else "RIGHT"   # turn away from it
            previous = self.last_steer.get(tr.id)
            if previous and abs(x_impact) < 0.01:          # heading near the centre:
                steer = previous                           # keep the earlier advice
            self.last_steer[tr.id] = steer
        return {"id": tr.id, "track": tr, "angle": angle, "x": x, "y": y, "close": close,
                "x_impact": x_impact, "closing": closing, "tti": tti, "level": level,
                "steer": steer, "blob": blob, "from_memory": from_memory, "vx": vx}

    def update(self, fused, radar_only, now):
        asteroids, live = [], set()
        for tr, blob, angle, from_memory in fused:
            a = np.radians(angle)
            x, y = tr.distance * np.sin(a), tr.distance * np.cos(a)
            vx = self._lateral_velocity(tr.id, x, now)
            asteroids.append(self._assess(tr, angle, x, y, vx, blob, from_memory))
            live.add(tr.id)
        for tr in radar_only:
            asteroids.append(self._assess(tr, None, None, tr.distance, 0.0, None, False))
            live.add(tr.id)
        for store in (self.lateral, self.last_steer):
            for track_id in list(store):
                if track_id not in live:
                    del store[track_id]
        asteroids.sort(key=lambda a: (THREAT_RANK[a["level"]],
                                      a["tti"] if a["tti"] is not None else np.inf,
                                      a["track"].distance))
        return asteroids


# ========================================================== route planner ====

ROUTE_COLOR = "#7df9ff"
SWEEP_PERIOD_S = 3.0          # one pass across the screen (left to right or back)
SWEEP_TRAIL_DEG = 40          # length of the fading trail behind the sweep line


class RoutePlanner:
    """Finds a safe route for the ship through the asteroids, weaving if needed.

    The space ahead is split into rows every 2 cm (forward) and columns every
    1 cm (sideways). A spot is safe if the ship (half its width plus a margin)
    can sit there without touching any asteroid. Each row the ship may slide
    up to 2 cm sideways (a 45 degree turn at most). Dynamic programming then
    finds the cheapest safe route from the ship to the far edge of the screen,
    where cost = how much it turns + how far it strays from its heading. So it
    keeps straight when it can, goes around when it must, and snakes back and
    forth through gaps when there's room - then comes back onto its heading.

    Only space the sensors can see is used (inside the camera's view), and
    asteroids with no known direction (radar only) can't be planned around.
    """

    DY, DX = 0.02, 0.01                 # row spacing (forward), column spacing (sideways)
    MAX_STEP = 2                        # columns per row: 2 cm per 2 cm = 45 degrees
    W_TURN, W_OFF, W_END, W_PREV = 1.0, 0.5, 0.5, 0.2

    def __init__(self, half_path, R, hfov_deg):
        self.half_path = half_path
        self.R = R
        self.ys = np.arange(self.DY, R + 1e-9, self.DY)
        self.xs = np.arange(-0.8 * R, 0.8 * R + 1e-9, self.DX)
        view = np.abs(self.xs)[None, :] <= (self.ys[:, None] * np.tan(np.radians(hfov_deg / 2))
                                            + half_path)
        self.view = view
        self.prev = None                 # last route's x per row (keeps it from flip-flopping)

    @staticmethod
    def obstacles(asteroids, hfov_deg):
        obs = []
        for a in asteroids:
            if a["x"] is None or a["track"].suspect:
                continue
            size = 0.03
            if a["blob"] is not None:   # width from the camera box
                size = a["track"].distance * np.radians(a["blob"]["box"][2] / CAM_W * hfov_deg)
            radius = float(np.clip(size / 2, 0.01, 0.05))
            radius += min(abs(a.get("vx", 0.0)) * 1.0, 0.03)   # sideways movers: extra room
            obs.append((a["x"], a["y"], radius, a["id"]))
        return obs

    def plan(self, obstacles):
        xs, ys, n = self.xs, self.ys, len(self.xs)
        free = self.view.copy()
        for ox, oy, r, _ in obstacles:
            free &= np.hypot(xs[None, :] - ox, ys[:, None] - oy) > self.half_path + r
        cost = np.full(n, np.inf)
        cost[np.argmin(np.abs(xs))] = 0.0             # start: the ship, on its heading
        back = np.zeros((len(ys), n), int)
        idx = np.arange(n)
        blocked_at = None
        prev = self.prev if self.prev is not None and len(self.prev) == len(ys) else None
        for k in range(len(ys)):
            best = np.full(n, np.inf)
            arg = np.zeros(n, int)
            for step in range(-self.MAX_STEP, self.MAX_STEP + 1):
                came = np.full(n, np.inf)
                if step >= 0:
                    came[step:] = cost[:n - step]
                else:
                    came[:step] = cost[-step:]
                c = came + self.W_TURN * (step * self.DX) ** 2
                better = c < best
                best[better] = c[better]
                arg[better] = idx[better] - step
            best += self.W_OFF * xs ** 2 * self.DY
            if prev is not None:
                best += self.W_PREV * (xs - prev[k]) ** 2 * self.DY
            best[~free[k]] = np.inf
            if not np.isfinite(best).any():
                blocked_at = k
                break
            cost, back[k] = best, arg
        last = (blocked_at if blocked_at is not None else len(ys)) - 1
        if last < 0:
            self.prev = None
            return {"path": [(0.0, 0.0)], "moves": [], "passes": [], "blocked_at": float(ys[0]),
                    "deviates": True}
        j = int(np.argmin(cost + self.W_END * xs ** 2))
        cols = [j]
        for k in range(last, 0, -1):
            j = back[k][j]
            cols.append(j)
        cols.reverse()
        path_x = xs[cols]
        path = [(0.0, 0.0)] + list(zip(path_x.tolist(), ys[:last + 1].tolist()))
        self.prev = None if blocked_at is not None else path_x
        # Which side the ship passes each nearby asteroid on, nearest first
        passes = []
        for ox, oy, r, oid in sorted(obstacles, key=lambda o: o[1]):
            if oy > ys[last]:
                continue
            px = float(np.interp(oy, ys[:last + 1], path_x))
            if abs(px - ox) <= self.half_path + r + 0.06:
                passes.append((oid, "LEFT" if px < ox else "RIGHT"))
        return {"path": path, "moves": self._moves(path), "passes": passes,
                "blocked_at": None if blocked_at is None else float(ys[blocked_at]),
                "deviates": bool(np.max(np.abs(path_x)) > self.DX / 2)}

    def _moves(self, path):
        """Route -> list of (move, y where it starts): HOLD / LEFT / RIGHT."""
        moves = []
        for (x0, y0), (x1, _) in zip(path[:-1], path[1:]):
            move = "HOLD" if abs(x1 - x0) < self.DX / 2 else ("LEFT" if x1 < x0 else "RIGHT")
            if moves and moves[-1][0] == move:
                continue
            moves.append((move, y0))
        # drop very short holds between turns the same way, and merge
        clean = []
        for i, (move, y) in enumerate(moves):
            end = moves[i + 1][1] if i + 1 < len(moves) else path[-1][1]
            if (move == "HOLD" and 0 < i < len(moves) - 1 and end - y < 0.05
                    and moves[i - 1][0] == moves[i + 1][0]):
                continue
            if clean and clean[-1][0] == move:
                continue
            clean.append((move, y))
        return clean


def route_text(route, long=False):
    """Short text for the banner, or the full route for the route line."""
    if route is None:
        return ""
    route.setdefault("passes", [])
    turns = [m for m, _ in route["moves"] if m != "HOLD"]
    if route["blocked_at"] is not None and not turns:
        return "NO GAP - REVERSE THRUST"
    if long:
        parts = [f"{m} @{y * 100:.0f}cm" for m, y in route["moves"] if m != "HOLD"][:4]
        if (route["blocked_at"] is None and route["moves"] and route["moves"][-1][0] == "HOLD"
                and abs(route["path"][-1][0]) < 0.015):
            parts.append("on course")
        text = " > ".join(parts) if parts else "HOLD"
        if route["blocked_at"] is not None:
            text += f" > BLOCKED @{route['blocked_at'] * 100:.0f}cm"
        return text
    if route["blocked_at"] is not None:
        return "NO GAP - REVERSE THRUST"
    if not turns:
        return "HOLD COURSE"
    sides = [side for _, side in route.get("passes", [])]
    weave = [s for i, s in enumerate(sides) if i == 0 or s != sides[i - 1]]
    if len(weave) >= 2:                  # passes asteroids on alternating sides
        return "SNAKE " + ">".join(t[0] for t in turns[:5])          # e.g. SNAKE L>R
    if len(turns) == 1:
        return f"STEER {turns[0]}"
    return f"GO AROUND {turns[0]}"                                   # out and back


def track_color(track_id):
    return TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


# ============================================================= display ====

class Display:
    def __init__(self, dist, hfov_deg, ship_width_m, heading_deg, warn_s, danger_s,
                 danger_m, warn_m, brace_m=0.05):
        self.brace_m = brace_m
        self.dist = dist
        self.R = float(dist[-1])
        self.hfov = hfov_deg
        self.half_path = ship_width_m / 2 + PATH_MARGIN
        self.heading = heading_deg % 360
        self.warn_s = warn_s
        self.danger_s = danger_s
        self.danger_m = danger_m
        self.warn_m = warn_m

        plt.style.use("dark_background")
        for key in list(plt.rcParams):
            if key.startswith("keymap."):
                plt.rcParams[key] = []
        plt.ion()

        self.fig = plt.figure(figsize=(14.5, 7.8))
        self.fig.canvas.manager.set_window_title("XE125 Spacecraft Debris Radar")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.3, 1],
                                   hspace=0.35, wspace=0.12, bottom=0.14, top=0.94)
        self.ax_ppi = self.fig.add_subplot(gs[:, 0])
        self.ax_cam = self.fig.add_subplot(gs[0, 1])
        self.ax_a = self.fig.add_subplot(gs[1, 1])
        self._setup_ppi()
        self._setup_camera()
        self._setup_ascope()
        self.status = self.fig.text(0.01, 0.015, "", color=GREEN, family="monospace", fontsize=10)
        self.tune_button = self._make_button([0.66, 0.008, 0.15, 0.05], "AUTO-TUNE: ON\n(click to pause)")
        self.reset_button = self._make_button([0.825, 0.008, 0.15, 0.05], "RESET\nSETTINGS")
        self.message = ""
        self.message_until = 0.0

    def _setup_ppi(self):
        ax, R = self.ax_ppi, self.R
        ax.set_aspect("equal")
        ax.set_xlim(-R * 1.05, R * 1.05)
        ax.set_ylim(-0.17 * R, R * 1.72)
        ax.axis("off")
        ax.set_title("SPACECRAFT DEBRIS RADAR", color=GREEN, fontsize=14, family="monospace")

        # Background stars (fixed pattern)
        rng = np.random.default_rng(7)
        sx = rng.uniform(-R * 1.05, R * 1.05, 260)
        sy = rng.uniform(-0.17 * R, R * 1.72, 260)
        ax.scatter(sx, sy, s=rng.uniform(0.5, 4, 260), color="white",
                   alpha=0.35, linewidths=0, zorder=0)

        # Range rings and bearing lines
        theta = np.linspace(0, np.pi, 200)
        for r in np.linspace(R / 4, R, 4):
            ax.plot(r * np.cos(theta), r * np.sin(theta), color=DIM_GREEN, lw=0.8, zorder=1)
            ax.text(r, -0.04 * R, f"{r * 100:.0f} cm", color=DIM_GREEN, ha="center",
                    fontsize=9, family="monospace")
        for deg in range(0, 181, 30):
            t = np.radians(deg)
            ax.plot([0, R * np.cos(t)], [0, R * np.sin(t)], color=DIM_GREEN, lw=0.5, zorder=1)
        # Compass bearings around the edge, relative to the ship's heading
        for rel in range(-90, 91, 30):
            t = np.radians(rel)
            y_lift = 0.06 * R if abs(rel) == 90 else 0.0   # keep clear of the range labels
            ax.text(R * 1.07 * np.sin(t), R * 1.07 * np.cos(t) + y_lift,
                    f"{(self.heading + rel) % 360:03.0f}°",
                    color=GREEN if rel == 0 else DIM_GREEN, ha="center", va="center",
                    fontsize=9, family="monospace", fontweight="bold" if rel == 0 else "normal")

        # Danger and warning distances
        for i, (radius, color, style, name) in enumerate(
                ((self.danger_m, RED, "-", "DANGER"), (self.warn_m, AMBER, "--", "WARNING"))):
            if radius < R:
                ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=color, lw=1.4,
                        ls=style, alpha=0.75, zorder=2)
            # small legend in the bottom-left corner
            ly = -0.06 * R - i * 0.045 * R
            ax.plot([-R, -0.92 * R], [ly, ly], color=color, lw=1.6, ls=style)
            ax.text(-0.9 * R, ly, f"{name} zone: {radius * 100:.0f} cm", color=color,
                    va="center", fontsize=8, family="monospace")
        ax.fill(np.r_[self.danger_m * np.cos(theta)], np.r_[self.danger_m * np.sin(theta)],
                color=RED, alpha=0.10, zorder=1)
        # Brace ring: solid white, with its own legend line
        ax.plot(self.brace_m * np.cos(theta), self.brace_m * np.sin(theta), color=WHITE,
                lw=1.6, alpha=0.9, zorder=2)
        ly = -0.06 * R - 2 * 0.045 * R
        ax.plot([-R, -0.92 * R], [ly, ly], color=WHITE, lw=1.6)
        ax.text(-0.9 * R, ly, f"BRACE zone: {self.brace_m * 100:.0f} cm", color=WHITE,
                va="center", fontsize=8, family="monospace")

        # Sensor field of view (camera)
        half = np.radians(self.hfov / 2)
        t = np.linspace(np.pi / 2 - half, np.pi / 2 + half, 60)
        ax.fill(np.r_[0, R * np.cos(t), 0], np.r_[0, R * np.sin(t), 0], color=GREEN, alpha=0.05,
                zorder=1)

        # The ship's path: a corridor straight ahead, as wide as the ship
        hw = self.half_path
        top = np.sqrt(max(R ** 2 - hw ** 2, 0))
        ax.fill([-hw, hw, hw, -hw], [0, 0, top, top], color="#00e5ff", alpha=0.08, zorder=1)
        ax.plot([-hw, -hw], [0, top], color="#00e5ff", lw=1, ls="--", alpha=0.6, zorder=2)
        ax.plot([hw, hw], [0, top], color="#00e5ff", lw=1, ls="--", alpha=0.6, zorder=2)
        ax.text(hw + 0.01 * R, 0.06 * R, "SHIP'S PATH", color="#00e5ff", fontsize=8,
                family="monospace", alpha=0.8, va="center")
        ax.annotate("", xy=(0, R * 0.98), xytext=(0, 0.02 * R), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color="#00e5ff", lw=1, alpha=0.5))

        # The spaceship (the XE125 sits in its nose)
        k = R
        ship = np.array([(0, 0), (0.022, -0.045), (0.065, -0.095), (0.022, -0.085),
                         (0.016, -0.115), (-0.016, -0.115), (-0.022, -0.085),
                         (-0.065, -0.095), (-0.022, -0.045)]) * k
        ax.fill(ship[:, 0], ship[:, 1], color="#cfd8dc", ec="white", lw=1, zorder=8)
        ax.fill(np.array([-0.012, 0.012, 0]) * k, np.array([-0.115, -0.115, -0.15]) * k,
                color=AMBER, alpha=0.9, zorder=7)
        ax.text(-0.09 * R, -0.08 * R, f"HDG {self.heading:03.0f}°", color="#00e5ff",
                ha="right", va="center", fontsize=10, family="monospace", fontweight="bold")

        lo, hi = np.radians(90 - BEAM_HALF_ANGLE_DEG), np.radians(90 + BEAM_HALF_ANGLE_DEG)
        self.arc_t = np.linspace(lo, hi, 60)
        self.dynamic = []
        self.banner = ax.text(0, R * 1.64, "", ha="center", va="center", fontsize=15,
                              family="monospace", fontweight="bold", zorder=12,
                              bbox=dict(facecolor=GREEN, edgecolor="none", pad=5))
        self.readout = ax.text(-R, R * 1.53, "", color=GREEN, fontsize=10, linespacing=1.15,
                               family="monospace", va="top", fontweight="bold", zorder=12)
        # Radar sweep: a bright line swinging back and forth across the screen
        # with a fading trail. (Visual only - the XE125 sees its whole beam at
        # once - but asteroids "ping" as the sweep passes over them.)
        self.sweep_on = True
        self.sweep_trail = []
        for i in range(8):
            poly = Polygon(np.zeros((3, 2)), closed=True, color=GREEN, lw=0,
                           alpha=0.22 * (1 - i / 8) ** 1.5, zorder=2)
            ax.add_patch(poly)
            self.sweep_trail.append(poly)
        self.sweep_line, = ax.plot([0, 0], [0, R], color=GREEN, lw=2.2, alpha=0.9, zorder=5)
        self.sweep_prev = None
        self.pings = {}                  # asteroid id -> time the sweep last crossed it
        self.route_line = ax.text(0.09 * R, -0.08 * R, "", color=ROUTE_COLOR, fontsize=8,
                                  family="monospace", fontweight="bold", va="center", zorder=12)

    def _setup_camera(self):
        ax = self.ax_cam
        ax.set_title("CAMERA  (draw the table zone: click points, or hold and trace)",
                     color=GREEN, fontsize=10, family="monospace")
        self.cam_img = ax.imshow(np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))
        ax.set_xticks([])
        ax.set_yticks([])
        # Shade everything OUTSIDE the table zone, so it's clear what's ignored
        self.outside_shade = ax.imshow(np.zeros((CAM_H, CAM_W, 4), dtype=np.float32), zorder=2)
        self.zone_line, = ax.plot([], [], color=AMBER, lw=2, ls="--", zorder=3)
        self.zone_label = ax.text(0, 0, "TABLE ZONE", color="black", fontsize=8,
                                  family="monospace", fontweight="bold", va="top", zorder=4,
                                  bbox=dict(facecolor=AMBER, edgecolor="none", pad=1.5))
        self.zone_flash_until = 0.0
        self.shown_zone = None
        # Outline being drawn: bright cyan, with a line to the mouse
        self.draw_line, = ax.plot([], [], color="#00e5ff", lw=2.5, marker="o", ms=4, zorder=9)
        self.draw_close, = ax.plot([], [], color="white", lw=1.2, ls=":", zorder=9)
        self.draw_text = ax.text(4, 4, "", color="black", fontsize=8, family="monospace",
                                 fontweight="bold", va="top", zorder=10, visible=False,
                                 bbox=dict(facecolor="#00e5ff", edgecolor="none", pad=2))
        self.boxes = []
        for _ in range(MAX_BLOBS):
            rect = Rectangle((0, 0), 0, 0, fill=False, lw=2, ec=GREEN, visible=False)
            ax.add_patch(rect)
            self.boxes.append(rect)
        self.box_labels = [ax.text(0, 0, "", fontsize=9, family="monospace", fontweight="bold",
                                   va="bottom") for _ in range(MAX_BLOBS)]

    def _make_button(self, rect, label):
        ax = self.fig.add_axes(rect)
        button = Button(ax, label, color="#0b3d0b", hovercolor="#1f7a12")
        button.label.set_color(GREEN)
        button.label.set_fontfamily("monospace")
        button.label.set_fontweight("bold")
        button.label.set_fontsize(9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREEN)
        return button

    def _setup_ascope(self):
        ax = self.ax_a
        ax.set_title("RADAR ECHO STRENGTH vs DISTANCE", color=GREEN, fontsize=10,
                     family="monospace")
        ax.set_xlabel("distance (cm)", color=DIM_GREEN)
        ax.set_xlim(self.dist[0] * 100, self.dist[-1] * 100)
        ax.tick_params(colors=DIM_GREEN)
        x = self.dist * 100
        self.amp_line, = ax.plot(x, np.zeros_like(x), color=GREEN, lw=1.2)
        self.thr_line, = ax.plot(x, np.zeros_like(x), color=AMBER, lw=1, ls="--")

    def flash(self, text, seconds=3.0):
        self.message = text
        self.message_until = time.time() + seconds

    def show_drawing(self, points, cursor, freehand):
        """Preview the outline being drawn."""
        path = list(points) + ([cursor] if cursor is not None and not freehand else [])
        self.draw_line.set_data([p[0] for p in path], [p[1] for p in path])
        self.draw_line.set_marker("" if freehand else "o")
        if len(points) >= 2:                       # dotted line showing how it will close
            end = cursor if (cursor is not None and not freehand) else points[-1]
            self.draw_close.set_data([end[0], points[0][0]], [end[1], points[0][1]])
        else:
            self.draw_close.set_data([], [])
        if freehand:
            hint = "tracing... release to finish"
        elif len(points) < 3:
            hint = f"point {len(points)}: keep clicking around the table"
        else:
            hint = f"{len(points)} points: double-click or Enter to finish"
        self.draw_text.set_text(hint)
        self.draw_text.set_visible(True)

    def hide_drawing(self):
        self.draw_line.set_data([], [])
        self.draw_close.set_data([], [])
        self.draw_text.set_visible(False)

    def _show_zone(self, cam_det):
        pts = cam_det.zone_points
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        self.zone_line.set_data(xs, ys)
        top = min(pts, key=lambda p: (p[1], p[0]))
        self.zone_label.set_position((min(max(top[0], 2), CAM_W - 70), top[1] + 2))
        shade = np.zeros((CAM_H, CAM_W, 4), dtype=np.float32)
        shade[~cam_det.zone, 3] = 0.55            # darken outside the zone
        self.outside_shade.set_data(shade)
        self.shown_zone = cam_det.zone_points

    def zone_was_set(self):
        self.zone_flash_until = time.time() + 1.5

    def update(self, frame, amp, threshold, asteroids, camera_only,
               radar_det, cam_det, collecting_bg, focus, auto_text="", route=None):
        # ---- camera view
        if frame is not None:
            self.cam_img.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.shown_zone is not cam_det.zone_points:
            self._show_zone(cam_det)
        just_set = time.time() < self.zone_flash_until
        self.zone_line.set_linestyle("-" if just_set else "--")   # solid for a moment after setting
        self.zone_line.set_linewidth(4 if just_set else 2)
        for rect, label in zip(self.boxes, self.box_labels):
            rect.set_visible(False)
            label.set_text("")
        shown = [(a["blob"], THREAT_COLORS[a["level"]], f"A{a['id']}")
                 for a in asteroids if a["blob"]]
        shown += [(blob, AMBER, "no radar") for blob in camera_only]
        for i, (blob, color, text) in enumerate(shown[:MAX_BLOBS]):
            x, y, w, h = blob["box"]
            self.boxes[i].set_bounds(x, y, w, h)
            self.boxes[i].set_edgecolor(color)
            self.boxes[i].set_visible(True)
            self.box_labels[i].set_position((x, y - 2))
            self.box_labels[i].set_text(text)
            self.box_labels[i].set_color(color)

        # ---- echo graph
        if amp is not None:
            self.amp_line.set_ydata(amp)
            thr = np.where(np.isfinite(threshold), threshold, np.nan)
            self.thr_line.set_ydata(thr)
            finite_thr = thr[np.isfinite(thr)]
            top = max(float(amp.max()) * 1.2,
                      float(finite_thr.max()) * 1.3 if finite_thr.size else 0, 1e-6)
            current = self.ax_a.get_ylim()[1]
            if top > current or top < current * 0.5:
                self.ax_a.set_ylim(0, top)

        # ---- radar screen: asteroids around the ship
        for artist in self.dynamic:
            artist.remove()
        self.dynamic = []
        ax, R = self.ax_ppi, self.R
        lines = []

        for ast in asteroids:
            tr, color = ast["track"], THREAT_COLORS[ast["level"]]
            tti_text = f"{ast['tti']:4.1f} s" if ast["tti"] is not None else "  -- "
            if ast["x"] is not None:
                x, y = ast["x"], ast["y"]
                faded = ast["from_memory"] or tr.coasting
                self.dynamic.append(ax.scatter([x], [y], s=430, marker=ASTEROID_SHAPE,
                                               color=color, alpha=0.45 if faded else 1.0,
                                               edgecolors="white", linewidths=1, zorder=9))
                # Predicted path to the ship's position (only if it's coming closer)
                if ast["tti"] is not None:
                    self.dynamic.append(ax.plot([x, ast["x_impact"]], [y, 0], color=color,
                                                lw=1.5, ls=":", alpha=0.8, zorder=4)[0])
                label = f"A{ast['id']}"
                if ast["tti"] is not None:
                    label += f"  {ast['tti']:.1f}s"
                if ast["from_memory"]:
                    label += " (camera lost)"
                self.dynamic.append(ax.text(x + 0.05 * R, y, label, color=color, fontsize=10,
                                            family="monospace", fontweight="bold",
                                            va="center", zorder=10))
                side = "R" if ast["angle"] > 0.5 else "L" if ast["angle"] < -0.5 else " "
                where = f"{abs(ast['angle']):4.1f}°{side}"
            else:                                  # radar only: distance, no bearing
                d = tr.distance
                self.dynamic.append(ax.plot(d * np.cos(self.arc_t), d * np.sin(self.arc_t),
                                            color=color, lw=3, alpha=0.6, zorder=4)[0])
                self.dynamic.append(ax.text(d * np.cos(self.arc_t[0]) + 0.02 * R,
                                            d * np.sin(self.arc_t[0]),
                                            f"A{ast['id']} " + ("unconfirmed?" if tr.suspect
                                                                 else "bearing unknown"),
                                            color=color,
                                            fontsize=9, family="monospace", zorder=10))
                where = "  ?   "
            if ast["tti"] is not None:
                speed = f"{ast['closing'] * 100:4.1f} cm/s"
            else:
                speed = "  not closing"[-12:]
            note = ast["level"]
            if tr.suspect:
                note = "UNCONFIRMED (near sensor, camera sees nothing)"
            elif ast["level"] == "BRACE":
                note = f"BRACE FOR IMPACT (within {self.brace_m * 100:.0f} cm)"
            elif ast["close"]:
                note = f"COLLISION (within {self.danger_m * 100:.0f} cm)"
            elif ast["level"] == "WARNING" and (ast["tti"] is None or ast["tti"] > self.warn_s):
                note = f"WARNING (within {self.warn_m * 100:.0f} cm)"
            elif ast["level"] == "CLEAR" and ast["tti"] is not None:
                note = f"CLEAR (passes {'right' if ast['x_impact'] > 0 else 'left'})"
            lines.append(f"A{ast['id']:<3}{tr.distance * 100:5.1f} cm {where}  {speed}  "
                         f"impact {tti_text}  {note}")

        for blob in camera_only:
            a = np.radians(blob["angle"])
            self.dynamic.append(ax.plot([0, R * np.sin(a)], [0, R * np.cos(a)], color=AMBER,
                                        lw=2, ls="--", alpha=0.8, zorder=4)[0])
            self.dynamic.append(ax.text(R * np.sin(a), R * np.cos(a) * 1.02, "no radar lock",
                                        color=AMBER, fontsize=9, family="monospace",
                                        ha="center", zorder=10))
            lines.append(f"?    ---       {abs(blob['angle']):4.1f}°   seen by camera, no radar lock")

        # Radar sweep and pings
        if self.sweep_on:
            now_s = time.monotonic()
            phase = (now_s / SWEEP_PERIOD_S) % 2.0           # 0..1 left->right, 1..2 back
            going_right = phase < 1.0
            frac = phase if going_right else 2.0 - phase
            sweep = np.radians(180.0 * (1 - frac))           # 180 deg = left edge, 0 = right
            self.sweep_line.set_data([0, R * np.cos(sweep)], [0, R * np.sin(sweep)])
            behind = 1.0 if going_right else -1.0            # trail lies behind the motion
            seg = np.radians(SWEEP_TRAIL_DEG) / len(self.sweep_trail)
            for i, poly in enumerate(self.sweep_trail):
                a0 = np.clip(sweep + behind * i * seg, 0, np.pi)
                a1 = np.clip(sweep + behind * (i + 1) * seg, 0, np.pi)
                t = np.linspace(a0, a1, 6)
                poly.set_xy(np.c_[np.r_[0, R * np.cos(t)], np.r_[0, R * np.sin(t)]])
            # an asteroid pings when the sweep line crosses its bearing
            if self.sweep_prev is not None:
                lo, hi = sorted((self.sweep_prev, sweep))
                for ast in asteroids:
                    if ast["x"] is not None:
                        bearing = np.arctan2(ast["y"], ast["x"])
                        if lo - 0.02 <= bearing <= hi + 0.02:
                            self.pings[ast["id"]] = now_s
            self.sweep_prev = sweep
            live = {a["id"] for a in asteroids}
            for aid in list(self.pings):
                if aid not in live:
                    del self.pings[aid]
            for ast in asteroids:
                age = now_s - self.pings.get(ast["id"], -99)
                if ast["x"] is not None and age < 0.8:
                    k = age / 0.8                                # 0 -> 1 as the ping fades
                    self.dynamic.append(ax.scatter(
                        [ast["x"]], [ast["y"]], s=430 + 2600 * k, facecolors="none",
                        edgecolors=THREAT_COLORS[ast["level"]], linewidths=2,
                        alpha=0.9 * (1 - k), zorder=8))

        # Planned route through the asteroids
        self.route_line.set_text("")
        if route is not None and radar_det.background is not None and (
                route["deviates"] or route["blocked_at"] is not None):
            px = np.array([p[0] for p in route["path"]])
            py = np.array([p[1] for p in route["path"]])
            if len(px) >= 5:                                   # gentle smoothing for drawing
                k = np.ones(3) / 3
                px = np.r_[px[0], np.convolve(px, k, "valid"), px[-1]]
            blocked = route["blocked_at"] is not None
            col = RED if blocked else ROUTE_COLOR
            # the ship's width along the route, then the route itself
            self.dynamic.append(ax.fill_betweenx(py, px - self.half_path, px + self.half_path,
                                                 color=col, alpha=0.10, zorder=3))
            self.dynamic.append(ax.plot(px, py, color=col, lw=3, alpha=0.95, zorder=6,
                                        solid_capstyle="round")[0])
            if len(px) >= 2:
                self.dynamic.append(ax.annotate(
                    "", xy=(px[-1], py[-1]), xytext=(px[-2], py[-2]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.8",
                                    color=col, lw=3)))
            if blocked:
                self.dynamic.append(ax.plot([px[-1]], [route["blocked_at"]], "x", color=RED,
                                            ms=16, mew=4, zorder=7)[0])
            for i, (move, y) in enumerate(route["moves"]):
                if move == "HOLD" or y <= 0.0 and i > 0:
                    continue
                kk = int(np.argmin(np.abs(py - y)))
                side = 1 if move == "LEFT" else -1      # label on the outside of the bend
                self.dynamic.append(ax.text(px[kk] + side * 0.03 * R, py[kk], move,
                                            color=col, fontsize=8, family="monospace",
                                            fontweight="bold", va="center",
                                            ha="right" if side < 0 else "left", zorder=11))
            self.route_line.set_text("ROUTE: " + route_text(route, long=True))
            self.route_line.set_color(col)

        # Alert banner + steering advice for the most urgent asteroid
        urgent = asteroids[0] if asteroids else None
        if radar_det.background is None:
            self.banner.set_text("")
            self.banner.get_bbox_patch().set_alpha(0)
        elif urgent and urgent["level"] == "BRACE":
            # Flashing red / white, three times a second
            blink = int(time.monotonic() * 3) % 2 == 0
            self.banner.set_text(f"!!! BRACE FOR IMPACT  -  A{urgent['id']} AT "
                                 f"{urgent['track'].distance * 100:.0f} cm !!!")
            self.banner.set_color(WHITE if blink else RED)
            self.banner.get_bbox_patch().set_facecolor(RED if blink else WHITE)
            self.banner.get_bbox_patch().set_alpha(1)
            half = np.linspace(0, np.pi, 60)
            self.dynamic.append(ax.fill(self.brace_m * 1.6 * np.cos(half),
                                        self.brace_m * 1.6 * np.sin(half),
                                        color=RED, alpha=0.55 if blink else 0.15,
                                        zorder=3)[0])
        elif urgent and urgent["level"] in ("COLLISION", "WARNING"):
            if route is not None and (route["deviates"] or route["blocked_at"] is not None):
                steer = route_text(route)
            else:
                steer = f"STEER {urgent['steer']}" if urgent["steer"] else "EVADE NOW"
            tti, dist_cm = urgent["tti"], urgent["track"].distance * 100
            if urgent["track"].suspect:
                text = (f"CHECK: A{urgent['id']} ECHO AT {dist_cm:.0f} cm  -  "
                        f"NOT SEEN BY CAMERA (possible ghost)")
            elif urgent["level"] == "COLLISION":
                if tti is not None and tti <= self.danger_s:
                    text = f"!! COLLISION IN {tti:.1f} s  -  A{urgent['id']}  -  {steer} !!"
                else:
                    text = f"!! IMMINENT COLLISION  -  A{urgent['id']} AT {dist_cm:.0f} cm  -  {steer} !!"
            elif tti is not None and tti <= self.warn_s:
                text = f"WARNING: A{urgent['id']} IMPACT IN {tti:.1f} s  -  {steer}"
            else:
                text = f"WARNING: A{urgent['id']} CLOSE AHEAD ({dist_cm:.0f} cm)  -  {steer}"
            color = THREAT_COLORS[urgent["level"]]
            self.banner.set_text(text)
            self.banner.set_color("black")
            self.banner.get_bbox_patch().set_facecolor(color)
            self.banner.get_bbox_patch().set_alpha(1)
            if urgent["steer"] and (route is None or urgent["x"] is None):
                sign = -1 if urgent["steer"] == "LEFT" else 1
                self.dynamic.append(ax.annotate(
                    "", xy=(sign * 0.32 * R, 0.16 * R), xytext=(0, 0.03 * R), zorder=11,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.9",
                                    color=color, lw=4,
                                    connectionstyle=f"arc3,rad={-0.35 * sign}")))
                self.dynamic.append(ax.text(sign * 0.34 * R, 0.2 * R, f"STEER {urgent['steer']}",
                                            color=color, fontsize=11, family="monospace",
                                            fontweight="bold", ha="center", zorder=11))
        else:
            in_path = [a for a in asteroids if a["level"] == "IN PATH"]
            text = (f"CAUTION: A{in_path[0]['id']} IN SHIP'S PATH" if in_path
                    else "ALL CLEAR")
            if in_path and route is not None and (route["deviates"] or route["blocked_at"] is not None):
                text += f"  -  {route_text(route)}"
            self.banner.set_text(text)
            self.banner.set_color("black")
            self.banner.get_bbox_patch().set_facecolor(YELLOW if in_path else GREEN)
            self.banner.get_bbox_patch().set_alpha(1)

        if radar_det.background is None:
            self.readout.set_text("")
        elif lines:
            shown_lines = lines[:4]                    # most urgent first
            if len(lines) > 4:
                shown_lines.append(f"... and {len(lines) - 4} more")
            self.readout.set_text("\n".join(shown_lines))
        else:
            self.readout.set_text("NO ASTEROIDS DETECTED")

        # ---- status line
        if collecting_bg == "record":
            msg = "RECORDING EMPTY SCENE - keep the table EMPTY and your hands away..."
        elif time.time() < self.message_until:
            msg = self.message
        elif radar_det.background is None or cam_det.background is None:
            msg = "Step 1: clear the table and press  b  to record the empty scene"
        else:
            msg = (f"{auto_text}    radar x{radar_det.factor:.1f}   focus {focus * 100:.0f}%   "
                   f"camera {cam_det.threshold:.0f}\n"
                   f"keys:  b empty scene   a auto-tune on/off   r reset   m mirror   q quit")
        self.status.set_text(msg)


# ================================================================ main ====

def parse_args():
    p = argparse.ArgumentParser(description="XE125 spacecraft debris radar (radar + webcam)")
    p.add_argument("--port", help="Serial port, e.g. COM3")
    p.add_argument("--mock", action="store_true", help="Use a simulated radar")
    p.add_argument("--far", action="store_true", help="Longer radar range (10 cm to 1.5 m)")
    p.add_argument("--camera", type=int, default=0, help="Camera number (default 0)")
    p.add_argument("--hfov", type=float, default=70.0,
                   help="Camera's horizontal field of view in degrees (default 70)")
    p.add_argument("--mirror", action="store_true", help="Flip left and right")
    p.add_argument("--focus", type=float, default=0.25,
                   help="Keep radar echoes at least this fraction as strong as the "
                        "strongest one (default 0.25)")
    p.add_argument("--objects", type=int, default=0,
                   help="Optional cap on how many objects to follow (default: no cap)")
    p.add_argument("--max-distance", type=float, default=None,
                   help="Ignore radar echoes beyond this distance, in cm")
    p.add_argument("--ship-width", type=float, default=10.0,
                   help="Width of the ship's path, in cm (default 10)")
    p.add_argument("--heading", type=float, default=0.0,
                   help="Ship's compass heading shown on screen, in degrees (default 0)")
    p.add_argument("--warn", type=float, default=10.0,
                   help="Warn when impact is less than this many seconds away (default 10)")
    p.add_argument("--danger", type=float, default=3.0,
                   help="Collision alert below this many seconds (default 3)")
    p.add_argument("--danger-distance", type=float, default=10.0,
                   help="Anything closer than this (cm) is an imminent collision (default 10)")
    p.add_argument("--no-sweep", action="store_true",
                   help="Turn off the sweeping line on the radar screen")
    p.add_argument("--boost", type=int, default=1, choices=[1, 2, 3, 4],
                   help="Average more on the radar chip for weaker objects: 2, 3 or 4 "
                        "times the normal averaging (default 1). Press b again after "
                        "changing it.")
    p.add_argument("--brace-distance", type=float, default=5.0,
                   help="Anything closer than this (cm) triggers BRACE FOR IMPACT (default 5)")
    p.add_argument("--near-band", type=float, default=0.33,
                   help="Fraction of the table zone's height, from the bottom (radar end), "
                        "where a camera object counts as close to the radar (default 0.33)")
    p.add_argument("--warn-distance", type=float, default=20.0,
                   help="Anything in the ship's path closer than this (cm) is a warning "
                        "(default 20)")
    return p.parse_args()


def open_client(args):
    if args.mock:
        return a121.Client.open(mock=True)
    if args.port:
        return a121.Client.open(serial_port=args.port)
    return a121.Client.open()


def main():
    args = parse_args()
    camera = Camera(args.camera)
    client = open_client(args)
    print("Radar connected:", client.server_info)
    # Try the requested boost; if the radar can't keep up at 10 frames per
    # second, step down until it can.
    for boost in [b for b in (args.boost, 3, 2, 1) if b <= args.boost]:
        config = make_config(args.far, boost)
        try:
            client.setup_session(config)
            break
        except Exception as exc:
            print(f"Boost x{boost} not accepted by the radar ({exc}); trying lower.")
    print(f"Radar averaging (HWAAS): {config.hwaas}")
    dist = distances_m(config)
    print(f"Radar measuring {dist[0] * 100:.0f} cm to {dist[-1] * 100:.0f} cm")

    radar_det = RadarDetector(dist, min_separation_m=0.10 if args.far else 0.05)
    tracker = Tracker()
    cam_det = CameraDetector()
    cam_det.max_objects = min(args.objects, MAX_BLOBS) if args.objects > 0 else MAX_BLOBS
    max_d = args.max_distance / 100 if args.max_distance else None
    display = Display(dist, args.hfov, args.ship_width / 100, args.heading,
                      args.warn, args.danger, args.danger_distance / 100,
                      args.warn_distance / 100, args.brace_distance / 100)
    if args.no_sweep:
        display.sweep_on = False
        display.sweep_line.set_visible(False)
        for poly in display.sweep_trail:
            poly.set_visible(False)
    assessor = ThreatAssessor(args.ship_width / 100, args.warn, args.danger,
                              args.danger_distance / 100, args.warn_distance / 100,
                              args.brace_distance / 100)
    planner = RoutePlanner(args.ship_width / 200 + PATH_MARGIN, float(dist[-1]), args.hfov)
    memory = {}
    state = {"running": True, "bg": None, "mirror": args.mirror,
             "draw": None,       # outline being drawn: points, held, freehand, press point
             "focus": min(max(args.focus, 0.0), 0.95),
             "phase": None,      # None or "record" (recording the empty scene)
             "auto": True,       # continuous auto-tune on/off
             "tune_job": None,   # background auto-tune thread
             "tune_result": None,
             "last_tune": 0.0,
             "tune_note": "waiting"}
    radar_buf = collections.deque(maxlen=TUNE_FRAMES)       # recent radar frames
    cam_buf = collections.deque(maxlen=TUNE_CAMERA_FRAMES)  # recent camera frames
    cam_buf_time = [0.0]

    def start_background():
        """b: record the EMPTY scene (nothing on the table)."""
        if state["phase"] is not None:
            return
        state["bg"] = []
        state["phase"] = "record"
        cam_det.start_background()

    def set_auto(on, why=""):
        state["auto"] = on
        state["last_tune"] = 0.0                   # tune soon after switching on
        display.tune_button.label.set_text("AUTO-TUNE: ON\n(click to pause)" if on
                                           else "AUTO-TUNE: OFF\n(click to resume)")
        display.flash(why or (f"Auto-tune {'on' if on else 'paused'}"), 4)

    def manual_change():
        """A manual adjustment pauses auto-tune, so it isn't undone a moment later."""
        if state["auto"]:
            set_auto(False, "Auto-tune paused for your manual change - press  a  to resume")

    def maybe_start_tune(now):
        """Start a background auto-tune when it's due and the scene allows it."""
        if (not state["auto"] or state["tune_job"] is not None or state["phase"] is not None
                or state["draw"] is not None or radar_det.background is None
                or len(radar_buf) < TUNE_FRAMES or now - state["last_tune"] < TUNE_EVERY_S):
            return
        # Wait while anything is moving: tuning on a moving object could hide it.
        if any(abs(tr.velocity) > MIN_CLOSING for tr in tracker.tracks if tr.confirmed):
            state["tune_note"] = "waiting (movement)"
            state["last_tune"] = now - TUNE_EVERY_S + 1.0    # look again in 1 s
            return
        state["last_tune"] = now
        radar_frames, cam_frames = list(radar_buf), list(cam_buf)

        def job():
            try:
                result = auto_tune(radar_frames, cam_frames, radar_det, cam_det)
            except Exception as exc:             # never let tuning crash the display
                print("auto-tune skipped:", exc)
                result = None
            state["tune_result"] = result if result is not None else "skip"
            state["tune_job"] = None

        state["tune_result"] = "pending"
        state["tune_job"] = threading.Thread(target=job, daemon=True)
        state["tune_job"].start()

    def blend(old, new):
        """Step toward a new value: most of the way if it's far off, halfway if close,
        so a bad setting is fixed quickly but small changes stay smooth."""
        far = new > 1.5 * old or new < old / 1.5
        weight = 0.8 if far else 0.5
        return (1 - weight) * old + weight * new

    def apply_tune_result():
        result = state["tune_result"]
        if result is None or result == "pending" or state["tune_job"] is not None:
            return
        state["tune_result"] = None
        if result == "skip":
            state["tune_note"] = "waiting (scene changing)"
            return
        if not state["auto"]:
            return
        radar_det.factor = blend(radar_det.factor, result["radar"])
        state["focus"] = blend(state["focus"], result["focus"])
        if result["camera"] is not None:
            cam_det.threshold = blend(cam_det.threshold, result["camera"])
        n = result["objects"]
        state["tune_note"] = f"tuned for {n} object{'s' if n != 1 else ''}"

    def reset_settings():
        state["tune_result"] = None
        radar_buf.clear()
        cam_buf.clear()
        radar_det.factor = DEFAULTS["radar"]
        cam_det.threshold = DEFAULTS["camera"]
        state["focus"] = DEFAULTS["focus"]
        if state["draw"] is not None:              # abandon an outline being drawn
            state["draw"] = None
            display.hide_drawing()
        cam_det.reset_zone()
        display.zone_was_set()                     # flash the restored zone outline
        set_auto(True, f"Reset to defaults: radar x{radar_det.factor:.1f}, camera "
                       f"{cam_det.threshold:.0f}, focus {state['focus'] * 100:.0f}%, table zone, "
                       f"auto-tune on. Clear the table and press  b .")

    def on_key(event):
        if event.key == "escape":
            cancel_drawing()
        elif event.key == "enter":
            finish_drawing()
        elif event.key == "b":
            start_background()
        elif event.key == "a":
            set_auto(not state["auto"])
        elif event.key == "r":
            reset_settings()
        elif event.key == "m":
            state["mirror"] = not state["mirror"]
            display.flash(f"Mirror {'on' if state['mirror'] else 'off'}")
        elif event.key == "up":
            manual_change()
            radar_det.factor *= 1.25
        elif event.key == "down":
            manual_change()
            radar_det.factor = max(1.5, radar_det.factor / 1.25)
        elif event.key == "]":
            manual_change()
            cam_det.threshold = min(cam_det.threshold + 5, 120)
        elif event.key == "[":
            manual_change()
            cam_det.threshold = max(cam_det.threshold - 5, 10)
        elif event.key == ".":
            manual_change()
            state["focus"] = min(round(state["focus"] + 0.05, 2), 0.95)
        elif event.key == ",":
            manual_change()
            state["focus"] = max(round(state["focus"] - 0.05, 2), 0.0)
        elif event.key == "q":
            state["running"] = False

    def clamp_to_image(x, y):
        return min(max(x, 0), CAM_W - 1), min(max(y, 0), CAM_H - 1)

    def mouse_in_camera(event):
        """Mouse position in camera pixels, kept inside the image."""
        if event.x is None or event.y is None:
            return None
        x, y = display.ax_cam.transData.inverted().transform((event.x, event.y))
        return clamp_to_image(x, y)

    def redraw_preview(cursor=None):
        d = state["draw"]
        display.show_drawing(d["points"], cursor, d["freehand"])
        display.fig.canvas.draw_idle()

    def on_press(event):
        """Left-click: add a point (or start tracing if the mouse is held and
        moved). Right-click: remove the last point. Double-click: finish."""
        if event.inaxes is not display.ax_cam or event.xdata is None:
            return
        toolbar = getattr(display.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            display.flash("Turn off the zoom / pan tool in the toolbar, then draw again")
            return
        point = clamp_to_image(event.xdata, event.ydata)
        d = state["draw"]
        if event.button == 3:                          # right-click: undo last point
            if d and d["points"]:
                d["points"].pop()
                if not d["points"]:
                    cancel_drawing()
                else:
                    redraw_preview(point)
            return
        if event.button != 1:
            return
        if d is None:                                  # start a new outline
            state["draw"] = {"points": [point], "held": True, "freehand": False,
                             "press": point}
            display.flash("Drawing the table zone: click points around the table, or hold "
                          "and trace. Double-click / Enter finishes, right-click undoes, "
                          "Esc cancels.", 60)
            redraw_preview(point)
            return
        if event.dblclick:                             # double-click: finish
            finish_drawing()
            return
        first = d["points"][0]
        if len(d["points"]) >= 3 and np.hypot(point[0] - first[0], point[1] - first[1]) < 8:
            finish_drawing()                           # clicked back on the first point
            return
        d["points"].append(point)
        d["held"] = True
        d["press"] = point
        redraw_preview(point)

    def on_motion(event):
        d = state["draw"]
        if d is None:
            return
        point = mouse_in_camera(event)
        if point is None:
            return
        if d["held"]:
            moved = np.hypot(point[0] - d["press"][0], point[1] - d["press"][1])
            if d["freehand"] or (moved > 6 and len(d["points"]) == 1):
                d["freehand"] = True                   # tracing with the button held
                last = d["points"][-1]
                if np.hypot(point[0] - last[0], point[1] - last[1]) >= 3:
                    d["points"].append(point)
        redraw_preview(point)

    def on_release(event):
        d = state["draw"]
        if d is None or event.button != 1:
            return
        d["held"] = False
        if d["freehand"]:
            finish_drawing()

    def finish_drawing():
        d = state["draw"]
        if d is None:
            return
        points = d["points"]
        # A double-click adds its own point twice; drop near-duplicates
        cleaned = [points[0]]
        for p in points[1:]:
            if np.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) >= 2:
                cleaned.append(p)
        state["draw"] = None
        display.hide_drawing()
        if len(cleaned) < 3 or not cam_det.set_zone(cleaned):
            display.flash("Zone too small - draw around a bigger area")
        else:
            display.zone_was_set()
            display.flash("Table zone set (outside is shaded). Clear the table and press  b .")
        display.fig.canvas.draw_idle()

    def cancel_drawing():
        if state["draw"] is not None:
            state["draw"] = None
            display.hide_drawing()
            display.flash("Drawing cancelled - the previous zone is kept")
            display.fig.canvas.draw_idle()

    canvas = display.fig.canvas
    canvas.mpl_connect("key_press_event", on_key)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("button_release_event", on_release)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("close_event", lambda e: state.update(running=False))
    display.tune_button.on_clicked(lambda _: set_auto(not state["auto"]))
    display.reset_button.on_clicked(lambda _: reset_settings())

    frames = queue.Queue()
    reader_error = []

    def read_radar():
        try:
            while state["running"]:
                result = client.get_next()
                frames.put((time.monotonic(), RadarDetector.frame_vector(result)))
        except Exception as exc:
            reader_error.append(exc)
            state["running"] = False

    client.start_session()
    reader = threading.Thread(target=read_radar, daemon=True)
    reader.start()
    try:
        amp = threshold = None
        tracks = []
        now = time.monotonic()
        while state["running"]:
            collecting = state["phase"]
            frame = camera.latest()

            while True:                               # every radar frame that arrived
                try:
                    now, vec = frames.get_nowait()
                except queue.Empty:
                    break
                if state["phase"] == "record":
                    state["bg"].append(vec)
                    if len(state["bg"]) >= BACKGROUND_FRAMES:
                        radar_det.set_background(state["bg"])
                        cam_det.finish_background()
                        state["bg"] = None
                        state["phase"] = None
                        tracker.reset()
                        assessor.reset()
                        planner.prev = None
                        memory.clear()
                        radar_buf.clear()
                        cam_buf.clear()
                        state["last_tune"] = 0.0
                        display.flash("Empty scene recorded. Place your objects - auto-tune "
                                      "adjusts the settings by itself.", 6)
                amp, threshold, detections = radar_det.update(vec)
                if radar_det.background is not None and state["phase"] is None:
                    radar_buf.append(amp)
                if max_d is not None:
                    detections = [det for det in detections if det[0] <= max_d]
                if radar_det.background is not None and state["phase"] != "record":
                    confirmed = tracker.update(detections, now)
                    tracks = focus_tracks(confirmed, state["focus"], args.objects)

            if frame is not None and collecting == "record":
                cam_det.add_background(frame)
            elif frame is not None and now - cam_buf_time[0] >= 0.5:
                cam_buf.append(frame)                # a camera frame every 0.5 s for tuning
                cam_buf_time[0] = now

            apply_tune_result()
            maybe_start_tune(now)
            if state["auto"]:
                auto_text = f"AUTO-TUNE ON ({state['tune_note']})"
            else:
                auto_text = "AUTO-TUNE OFF"

            blobs = []
            if frame is not None and collecting != "record":
                for blob in cam_det.find(frame):
                    blob["angle"] = pixel_to_angle_deg(blob["centre_x"], args.hfov, state["mirror"])
                    blobs.append(blob)

            if radar_det.background is not None and collecting != "record":
                near_row = near_band_limit(cam_det.zone_points, args.near_band)
                fused, radar_only, camera_only = fuse(
                    tracks, blobs, memory, now, near_row=near_row,
                    camera_ok=frame is not None and cam_det.background is not None)
            else:
                fused, radar_only, camera_only = [], [], []
            asteroids = assessor.update(fused, radar_only, now)
            route = None
            if radar_det.background is not None and collecting != "record":
                route = planner.plan(RoutePlanner.obstacles(asteroids, args.hfov))
            display.update(frame, amp, threshold, asteroids, camera_only,
                           radar_det, cam_det, collecting, state["focus"], auto_text, route)
            canvas.draw_idle()
            plt.pause(1 / DISPLAY_FPS)
    finally:
        state["running"] = False
        reader.join(timeout=2)
        client.stop_session()
        client.close()
        camera.close()
        print("Disconnected.")
        if reader_error:
            raise reader_error[0]


if __name__ == "__main__":
    main()
