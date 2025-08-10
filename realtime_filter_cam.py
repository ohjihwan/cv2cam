import cv2
import numpy as np
from datetime import datetime
import time
import os

# Haar 얼굴 검출기 로드
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 선글라스 PNG 로드 (동일 폴더)
SG_PATH = os.path.join(os.path.dirname(__file__), 'sunglasses.png')
SG_IMG = cv2.imread(SG_PATH, cv2.IMREAD_UNCHANGED)
if SG_IMG is None:
	print("⚠️ 'sunglasses.png' 를 찾을 수 없습니다. 같은 폴더에 두었는지 확인하세요.")
else:
	# 알파 채널이 없으면 추가(불투명)
	if SG_IMG.shape[2] == 3:
		alpha = np.full(SG_IMG.shape[:2] + (1,), 255, dtype=np.uint8)
		SG_IMG = np.concatenate([SG_IMG, alpha], axis=2)

# --------------------------
# 카메라 자동 스캔 (0~4)
# --------------------------
def open_any_camera(preferred_index=0, try_range=5):
	tried = []
	for idx in [preferred_index] + [i for i in range(try_range) if i != preferred_index]:
		cap = cv2.VideoCapture(idx)
		if not cap.isOpened():
			tried.append(idx)
			continue
		ok, _ = cap.read()
		if ok:
			return cap, idx
		cap.release()
		tried.append(idx)
	raise RuntimeError(f"카메라를 열 수 없습니다. 시도한 인덱스: {tried}")

# --------------------------
# 알파블렌딩(스티커 합성)
# --------------------------
def alpha_blend_rgba(frame_bgr, sticker_bgra, center_xy):
	x, y = center_xy
	H, W = frame_bgr.shape[:2]
	sh, sw = sticker_bgra.shape[:2]

	x0 = int(x - sw // 2)
	y0 = int(y - sh // 2)

	x1 = max(0, x0); y1 = max(0, y0)
	x2 = min(W, x0 + sw); y2 = min(H, y0 + sh)
	if x1 >= x2 or y1 >= y2:
		return frame_bgr

	sx1 = x1 - x0; sy1 = y1 - y0
	sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)

	roi = frame_bgr[y1:y2, x1:x2]
	sticker_crop = sticker_bgra[sy1:sy2, sx1:sx2]
	sticker_rgb = sticker_crop[:, :, :3]
	alpha = sticker_crop[:, :, 3:4] / 255.0
	blended = (alpha * sticker_rgb + (1 - alpha) * roi).astype(np.uint8)
	frame_bgr[y1:y2, x1:x2] = blended
	return frame_bgr

# --------------------------
# 스티커(BGRA) 생성기
# --------------------------
def make_heart(size):
	s = size
	canvas = np.zeros((s, s, 4), np.uint8)
	red = (0, 0, 255, 255)
	overlay = np.zeros_like(canvas)
	cv2.circle(overlay, (s//3, s//3), s//4, red, -1)
	cv2.circle(overlay, (2*s//3, s//3), s//4, red, -1)
	pts = np.array([[s//6, s//3], [5*s//6, s//3], [s//2, s - s//8]], np.int32)
	cv2.fillPoly(overlay, [pts], red)
	mask = overlay[:, :, 2] > 0
	canvas[mask] = overlay[mask]
	return canvas

def make_star(size):
	s = size
	canvas = np.zeros((s, s, 4), np.uint8)
	yellow = (0, 255, 255, 255)
	cx, cy = s//2, s//2
	r1, r2 = int(s*0.45), int(s*0.18)
	pts = []
	for i in range(10):
		ang = np.deg2rad(i * 36 - 90)
		r = r1 if i % 2 == 0 else r2
		px = int(cx + r * np.cos(ang))
		py = int(cy + r * np.sin(ang))
		pts.append([px, py])
	cv2.fillPoly(canvas, [np.array(pts, np.int32)], yellow)
	return canvas

def make_smile(size):
	s = size
	canvas = np.zeros((s, s, 4), np.uint8)
	face = (0, 255, 255, 255)
	black = (0, 0, 0, 255)
	cv2.circle(canvas, (s//2, s//2), s//2 - 2, face, -1)
	cv2.circle(canvas, (int(s*0.35), int(s*0.4)), s//12, black, -1)
	cv2.circle(canvas, (int(s*0.65), int(s*0.4)), s//12, black, -1)
	cv2.ellipse(canvas, (s//2, int(s*0.6)), (int(s*0.25), int(s*0.18)), 0, 20, 160, black, thickness=max(2, s//16))
	return canvas

def build_sticker(kind, size):
	return make_heart(size) if kind == 1 else make_star(size) if kind == 2 else make_smile(size)

# --------------------------
# 필터
# --------------------------
def apply_filter(frame, mode):
	if mode == 'none': return frame
	if mode == 'gray':
		g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
	if mode == 'blur':
		return cv2.GaussianBlur(frame, (9, 9), 0)
	if mode == 'edge':
		g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(g, 80, 150)
		return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
	if mode == 'sepia':
		kernel = np.array([[0.272, 0.534, 0.131],
						   [0.349, 0.686, 0.168],
						   [0.393, 0.769, 0.189]], np.float32)
		return np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)
	if mode == 'cartoon':
		color = cv2.bilateralFilter(frame, 9, 150, 150)
		g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
									  cv2.THRESH_BINARY, 9, 9)
		edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
		return cv2.bitwise_and(color, edges)
	return frame

HELP_TEXT = [
	"Controls:",
	"  Filters: [0] none, [g] gray, [b] blur, [e] edge, [s] sepia, [c] cartoon",
	"  Stickers: [1] heart, [2] star, [3] smile",
	"  Place: mouse LEFT click (multiple)",
	"  Size: [+] bigger, [-] smaller",
	"  Clear stickers: [4]",
	"  Face filter: [f] sunglasses",
	"  Quit & Save: [ESC]",
]

def put_help(frame, filter_mode, sticker_kind, scale, cam_idx, face_on=False):
	overlay = frame.copy()
	h = 10 + 22*(len(HELP_TEXT)+4)
	cv2.rectangle(overlay, (10, 10), (440, h), (0,0,0), -1)
	frame[:] = (0.45*overlay + 0.55*frame).astype(np.uint8)
	y = 24
	for line in HELP_TEXT:
		cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
		y += 22
	cv2.putText(frame, f"Filter: {filter_mode}", (20, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120), 2, cv2.LINE_AA)
	cv2.putText(frame, f"Sticker: { {1:'heart',2:'star',3:'smile'}[sticker_kind] } | scale: {scale:.2f}x | cam:{cam_idx}",
				(20, y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120), 2, cv2.LINE_AA)
	cv2.putText(frame, f"Face filter: {'ON' if face_on else 'OFF'}",
				(20, y+56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120), 2, cv2.LINE_AA)

def main():
	try:
		cap, cam_idx = open_any_camera(0, 5)
	except RuntimeError as e:
		print("⚠️", e)
		print("macOS 카메라 권한 확인: 시스템 설정 → 개인정보 보호 및 보안 → 카메라 → Terminal/Python 체크")
		return

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	# 저장 헤더 FPS는 안전하게 30으로 고정 (재생속도 기준)
	header_fps = 30.0

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out_name = f"opencv_cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
	writer = None
	last_size = None

	# 프레임 페이싱용 타임스탬프/카운터
	t0 = None
	frames_written = 0

	filter_mode = 'none'
	sticker_kind = 1
	sticker_scale = 1.0
	placed = []

	face_filter_on = False
	frame_idx = 0

	# ---- 얼굴 박스 안정화 상태 ----
	DETECT_INTERVAL = 3		# 매 N프레임마다 검출
	SMOOTH_ALPHA = 0.25		# 지수평활 계수(0.2~0.35 추천)
	MAX_MISS = 8			# 미검출 허용 프레임(이내면 마지막 박스 유지)
	face_ok = False
	fx = fy = fw = fh = 0.0
	miss = 0

	win = "OpenCV Real-time Filter Cam"
	cv2.namedWindow(win)

	def on_mouse(event, x, y, flags, param):
		nonlocal sticker_kind, sticker_scale, placed
		if event == cv2.EVENT_LBUTTONDOWN:
			H = param.get('height', 480)
			base = int(max(40, H * 0.15))
			size = max(20, int(base * sticker_scale))
			placed.append((sticker_kind, (x, y), size))
	cv2.setMouseCallback(win, on_mouse, param={'height': 480})

	print("\n[조작법]")
	for t in HELP_TEXT:
		print(" -", t)
	print("실행 중… ESC로 종료하면 동영상이 저장됩니다.\n")

	while True:
		ok, frame = cap.read()
		if not ok:
			print("⚠️ 프레임을 읽지 못했습니다. 카메라 상태/권한을 확인하세요.")
			break

		H, W = frame.shape[:2]
		cv2.setMouseCallback(win, on_mouse, param={'height': H})

		# writer(크기/헤더 FPS) 준비 및 타임기준 초기화
		if last_size != (W, H):
			if writer is not None:
				writer.release()
			writer = cv2.VideoWriter(out_name, fourcc, header_fps, (W, H))
			last_size = (W, H)
			t0 = time.time()
			frames_written = 0

		processed = apply_filter(frame, filter_mode)

		# ---------- 얼굴 선글라스 합성 (안정화) ----------
		frame_idx += 1
		if face_filter_on and SG_IMG is not None and SG_IMG.size != 0:
			do_detect = (frame_idx % DETECT_INTERVAL == 0) or (not face_ok)
			if do_detect:
				gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
				faces = FACE_CASCADE.detectMultiScale(
					gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
				)
				if len(faces) > 0:
					areas = [(w*h, (x, y, w, h)) for (x, y, w, h) in faces]
					_, (nx, ny, nw, nh) = max(areas, key=lambda t: t[0])
					if face_ok:
						fx = SMOOTH_ALPHA*nx + (1-SMOOTH_ALPHA)*fx
						fy = SMOOTH_ALPHA*ny + (1-SMOOTH_ALPHA)*fy
						fw = SMOOTH_ALPHA*nw + (1-SMOOTH_ALPHA)*fw
						fh = SMOOTH_ALPHA*nh + (1-SMOOTH_ALPHA)*fh
					else:
						fx, fy, fw, fh = float(nx), float(ny), float(nw), float(nh)
					face_ok = True
					miss = 0
				else:
					if face_ok:
						miss += 1
						if miss > MAX_MISS:
							face_ok = False
					else:
						face_ok = False

			if face_ok:
				sg_w = int(fw * 1.15)
				ratio = SG_IMG.shape[0] / SG_IMG.shape[1]
				sg_h = int(sg_w * ratio)
				cx = int(fx + fw / 2)
				cy = int(fy + fh * 0.42)  # 눈 위치
				sg_resized = cv2.resize(SG_IMG, (max(1, sg_w), max(1, sg_h)), interpolation=cv2.INTER_AREA)
				processed = alpha_blend_rgba(processed, sg_resized, (cx, cy))

		# 스티커 누적 합성
		for k, (sx, sy), sz in placed:
			sticker = build_sticker(k, sz)
			processed = alpha_blend_rgba(processed, sticker, (sx, sy))

		put_help(processed, filter_mode, sticker_kind, sticker_scale, cam_idx, face_on=face_filter_on)
		cv2.imshow(win, processed)

		# ---- 프레임 페이싱: 벽시계 기준으로 필요한 만큼 기록 ----
		if writer is not None and t0 is not None:
			now = time.time()
			expected = int((now - t0) * header_fps)  # 지금까지 들어가야 하는 총 프레임 수
			while frames_written < expected:
				writer.write(processed)			# 마지막 처리 프레임을 중복 기록해 타이밍 유지
				frames_written += 1

		# ---- 키 입력 처리 (mac 친화 / 대소문자 & 넘패드 허용) ----
		key = cv2.waitKeyEx(10)
		if key == -1:
			continue

		def is_key(*codes):
			return key in codes

		if is_key(27):  # ESC
			break
		elif is_key(ord('0'), 48):
			filter_mode = 'none'
		elif is_key(ord('g'), ord('G')):
			filter_mode = 'gray'
		elif is_key(ord('b'), ord('B')):
			filter_mode = 'blur'
		elif is_key(ord('e'), ord('E')):
			filter_mode = 'edge'
		elif is_key(ord('s'), ord('S')):
			filter_mode = 'sepia'
		elif is_key(ord('c'), ord('C')):
			filter_mode = 'cartoon'
		elif is_key(ord('1'), 49, 65457):
			sticker_kind = 1
		elif is_key(ord('2'), 50, 65458):
			sticker_kind = 2
		elif is_key(ord('3'), 51, 65459):
			sticker_kind = 3
		elif is_key(ord('4'), 52, 65460):
			placed.clear()
		elif is_key(ord('f'), ord('F')):
			face_filter_on = not face_filter_on
			if not face_filter_on:
				face_ok = False; miss = 0
			# ⚠️ 페이싱은 계속 유지되므로, f 토글에 따라 재생 속도는 변하지 않음
		elif is_key(ord('+'), ord('='), 43, 61):
			sticker_scale = min(3.0, sticker_scale + 0.1)
		elif is_key(ord('-'), ord('_'), 45, 95):
			sticker_scale = max(0.3, sticker_scale - 0.1)

	cap.release()
	if writer is not None:
		# 마지막으로 누락된 프레임 있으면 채움
		now = time.time()
		expected = int((now - t0) * header_fps) if t0 is not None else 0
		while frames_written < expected:
			writer.write(processed)
			frames_written += 1
		writer.release()
	cv2.destroyAllWindows()
	print(f"\n✅ 저장 완료: {out_name}")

if __name__ == "__main__":
	main()
