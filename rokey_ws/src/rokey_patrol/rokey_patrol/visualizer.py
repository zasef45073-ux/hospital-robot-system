#!/usr/bin/env python3
"""
================================================================================
  사람/침대 시각화 모듈 (디버그용 OpenCV 창 그리기)
================================================================================

[전체 개요]
이 모듈은 amr_detect.py 노드에서 카메라 프레임 위에 검출 결과를 오버레이로 그려
별도의 OpenCV 창에 띄우는 역할을 합니다. 로봇이 "지금 무엇을 보고 무슨 판단을
하는 중인지"를 사람이 눈으로 확인할 수 있도록 해주는 디버그 화면입니다.

[화면에 그려지는 것들]
  1) 사람 박스 (자세별 색깔):
       - Standing(초록), Sitting(주황), Lying(빨강), Unknown(회색) 등
  2) 사람 골격(스켈레톤): 17개 키포인트 중 어깨~다리 라인 연결선
  3) 침대 마커(ArUco 등): 좌/우/중간 마커 위치와 침대 후보 영역(반투명 사각형)
  4) 텍스트 오버레이: PERSON 상태, MODE, ALIGN 정보, POSE, YAW, PROCESSED 카운트
  5) 가이드 라인: 화면 중앙선 + 좌우 가장자리 마진선

[좌표계 주의]
OpenCV 이미지는 (y행, x열) 순서로 shape이 나오지만, 그리기 함수는 (x, y) 순서를 씀.
- frame.shape[:2] = (h, w)
- cv2.line(img, (x1, y1), (x2, y2), ...)

[색상 형식 주의]
OpenCV는 RGB가 아니라 BGR 순서임. 즉 (0, 0, 255) 가 빨강이고 (0, 255, 0) 이 초록.
"""

import math

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 자세별 색상 매핑 (BGR 순서)
#   Standing    : 초록   - 안전, "정상 상태"
#   Sitting     : 주황   - 안전이지만 주의
#   Lying       : 빨강   - 위험 가능성, 주목해야 할 자세
#   LyingInBed  : 보라   - 침대 위에 누움 (정상일 가능성 높음)
#   LyingOutOfBed: 빨강  - 침대 밖 누움 (실제 위험)
#   Unknown     : 회색   - 판단 불가
# ─────────────────────────────────────────────────────────────────────────────
POSE_COLOR = {
    'Standing':      (0, 255, 0),
    'Sitting':       (0, 165, 255),
    'Lying':         (0, 0, 255),
    'LyingInBed':    (255, 0, 255),
    'LyingOutOfBed': (0, 0, 255),
    'Unknown':       (200, 200, 200),
}

# ─────────────────────────────────────────────────────────────────────────────
# 골격(스켈레톤) 연결 정보
#   COCO 17 keypoint 인덱스 기준으로 "어떤 점과 어떤 점을 선으로 잇는지" 정의.
#
#   키포인트 인덱스 참고:
#     0  nose,        1  left_eye,    2  right_eye,
#     3  left_ear,    4  right_ear,
#     5  left_shoulder,  6 right_shoulder,
#     7  left_elbow,     8 right_elbow,
#     9  left_wrist,    10 right_wrist,
#     11 left_hip,      12 right_hip,
#     13 left_knee,     14 right_knee,
#     15 left_ankle,    16 right_ankle
#
#   여기서는 얼굴 라인(0~4)은 그리지 않고 어깨 아래 몸통/팔/다리만 연결.
#   → 얼굴 점들은 따로 점으로만 찍힘 (가독성 위해)
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_EDGES = [
    (5, 6),                     # 좌우 어깨 연결 (가슴 라인)
    (5, 7), (7, 9),             # 왼팔: 어깨-팔꿈치-손목
    (6, 8), (8, 10),            # 오른팔: 어깨-팔꿈치-손목
    (5, 11), (6, 12),           # 어깨에서 골반으로 내려가는 몸통 라인
    (11, 12),                   # 좌우 골반 연결 (허리 라인)
    (11, 13), (13, 15),         # 왼다리: 골반-무릎-발목
    (12, 14), (14, 16),         # 오른다리: 골반-무릎-발목
]


# 자주 쓰는 폰트/색 상수 (모듈 단위로 미리 정의해두면 매번 생성 안 해도 됨)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRAY = (200, 200, 200)


# ─────────────────────────────────────────────────────────────────────────────
# cv2.putText 호출용 짧은 헬퍼
#   cv2.putText는 인자가 너무 많아서 호출이 길어지므로 자주 쓰는 형태를 함수로 묶음.
#   img    : 그릴 이미지
#   text   : 표시할 문자열
#   pos    : (x, y) 좌측-하단 기준 좌표
#   scale  : 글자 크기 배율 (예: 0.5)
#   color  : BGR 색
#   thickness: 글자 두께 (기본 2)
# ─────────────────────────────────────────────────────────────────────────────
def _put(img, text, pos, scale, color, thickness=2):
    cv2.putText(img, text, pos, _FONT, scale, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# PersonVisualizer
#   한 프레임에 대해 모든 시각화를 묶어서 그려주는 클래스.
#   외부에서는 render(...) 한 번만 호출하면 끝나도록 인터페이스를 단일화.
#
#   주요 파라미터:
#     window_name    : OpenCV 창 제목
#     kpt_conf_th    : 키포인트 신뢰도가 이 값 이상일 때만 그림 (낮은 점은 노이즈)
#     edge_margin_px : "가장자리 가이드 선" 위치 (사람이 화면 끝에 붙었는지 보는 기준)
#     display_scale  : 화면에 띄울 때 몇 배로 확대할지 (예: 2.0이면 두 배 크기로 띄움)
# ─────────────────────────────────────────────────────────────────────────────
class PersonVisualizer:
    def __init__(
        self,
        window_name='rokey Person Align',
        kpt_conf_th=0.3,
        edge_margin_px=120,
        display_scale=2.0,
    ):
        self.window_name = window_name
        self.kpt_conf_th = kpt_conf_th
        self.edge_margin_px = edge_margin_px
        self.display_scale = display_scale

    # ─────────────────────────────────────────────────────────────────
    # 한 사람의 골격(스켈레톤)을 그리는 함수
    #   keypoints: shape (N, 3) 배열. 각 행 = (x, y, confidence)
    #   color    : 선의 색
    #
    #   1) SKELETON_EDGES 에 정의된 점쌍을 순회하며 양 끝점이 모두 신뢰도 이상이면 선 연결
    #   2) 모든 키포인트 중 신뢰도 이상인 점은 파란색 동그라미로 표시
    # ─────────────────────────────────────────────────────────────────
    def _draw_skeleton(self, image, keypoints, color):
        if keypoints is None:
            return

        n = len(keypoints)
        th = self.kpt_conf_th

        # 1) 선(에지) 그리기
        for a, b in SKELETON_EDGES:
            # 인덱스가 키포인트 길이를 벗어나면 안전하게 스킵
            if a >= n or b >= n:
                continue

            xa, ya, ca = keypoints[a]
            xb, yb, cb = keypoints[b]

            # 양 끝 점 모두 신뢰도가 임계값 이상일 때만 연결선 그림 (한쪽만 신뢰 가능하면 안 그림)
            if ca >= th and cb >= th:
                cv2.line(image,
                         (int(xa), int(ya)),
                         (int(xb), int(yb)),
                         color, 2)

        # 2) 점(키포인트) 그리기 - 모든 키포인트 중 신뢰도 이상인 것만
        for x, y, conf in keypoints:
            if conf >= th:
                # (255, 0, 0) = 파란색 (BGR), -1 = 안을 채움
                cv2.circle(image, (int(x), int(y)), 4, (255, 0, 0), -1)

    # ─────────────────────────────────────────────────────────────────
    # 검출된 사람들의 박스 + 스켈레톤 + 라벨 그리기
    #   target_index 와 매칭되는 사람은 "현재 타겟"으로 강조 표시:
    #     - 박스가 더 두껍게 (3 vs 2)
    #     - 스켈레톤 색이 노란색(0, 255, 255)
    #     - 라벨에 "TARGET" 접두어
    # ─────────────────────────────────────────────────────────────────
    def _draw_people(self, image, people, target_index):
        for p in people:
            # 박스 좌표 - 정수로 변환해야 OpenCV가 받음
            x1, y1, x2, y2 = map(int, p['box'])
            pose = p.get('pose', 'Unknown')
            is_target = p.get('index') == target_index
            color = POSE_COLOR.get(pose, _GRAY)

            # 박스: 타겟이면 두께 3, 아니면 2
            cv2.rectangle(image, (x1, y1), (x2, y2),
                          color, 3 if is_target else 2)

            # 스켈레톤 색: 타겟이면 시안(노랑끼), 아니면 어두운 olive 톤
            self._draw_skeleton(
                image,
                p.get('keypoints'),
                color=(0, 255, 255) if is_target else (180, 180, 0),
            )

            # 라벨 예: "TARGET Lying 0.87"
            label = f"{'TARGET ' if is_target else ''}{pose} {p.get('conf', 0.0):.2f}"
            # y - 10 자리에 쓰되, 너무 위로 가서 잘리지 않게 최소 y=20 으로 클램프
            _put(image, label, (x1, max(y1 - 10, 20)), 0.53, color)

    # ─────────────────────────────────────────────────────────────────
    # 침대 마커 형식 통일 헬퍼
    #   외부에서 마커를 dict({'center': (x, y), ...}) 또는 그냥 (x, y) 튜플로
    #   넘길 수 있는데, 이 함수가 둘 다 받아서 (x, y) 정수 튜플로 정규화해줌.
    #   None 이거나 형식이 이상하면 None 반환.
    # ─────────────────────────────────────────────────────────────────
    def _marker_center(self, marker):
        if marker is None:
            return None

        # dict 형식: {'center': (x, y), ...}
        if isinstance(marker, dict):
            center = marker.get('center')
            if center is None:
                return None
            return int(center[0]), int(center[1])

        # 튜플/리스트 형식: (x, y) 또는 [x, y, ...]
        if isinstance(marker, (tuple, list)) and len(marker) >= 2:
            return int(marker[0]), int(marker[1])

        return None

    # ─────────────────────────────────────────────────────────────────
    # 침대 마커 정보 시각화
    #
    #   bed_markers      : 검출된 모든 마커 (M1, M2, ... 로 번호 매겨 표시)
    #   bed_left_marker  : 침대 왼쪽 경계 마커 (세로 보라색 선)
    #   bed_right_marker : 침대 오른쪽 경계 마커 (세로 보라색 선)
    #   bed_mid_marker   : 침대 깊이 기준 마커 (가로 주황색 선)
    #   bed_decision     : 'IN_BED' / 'OUT_OF_BED' / 'NO_MARKER' / 'UNKNOWN'
    #
    #   좌/우/중간 세 마커가 모두 잡히면, 그 사이 영역(중간선 위쪽)을
    #   "침대 후보 영역"으로 보고 반투명 보라색으로 칠함.
    # ─────────────────────────────────────────────────────────────────
    def _draw_bed_marker_info(
        self,
        image,
        bed_markers=None,
        bed_left_marker=None,
        bed_right_marker=None,
        bed_mid_marker=None,
        bed_decision='UNKNOWN',
    ):
        bed_markers = bed_markers or []
        h, w = image.shape[:2]

        # ── 검출된 모든 마커 표시: 노란 점 + 검은 외곽선 + M1, M2... 라벨 ──
        for i, marker in enumerate(bed_markers):
            center = self._marker_center(marker)
            if center is None:
                continue

            cx, cy = center
            cv2.circle(image, (cx, cy), 7, (0, 255, 255), -1)   # 채운 노란 원
            cv2.circle(image, (cx, cy), 10, (0, 0, 0), 2)       # 까만 외곽선 (가독성)
            _put(image, f'M{i + 1}', (cx + 8, cy - 8), 0.42, (0, 255, 255), 2)

        # 좌/우/중간 마커 위치를 정규화
        left  = self._marker_center(bed_left_marker)
        right = self._marker_center(bed_right_marker)
        mid   = self._marker_center(bed_mid_marker)

        # ── 좌측 경계: 마커 x를 가로지르는 세로 보라색 선 + 표식 ──
        if left is not None:
            lx, ly = left
            cv2.line(image, (lx, 0), (lx, h), (255, 0, 255), 2)
            cv2.circle(image, (lx, ly), 11, (255, 0, 255), 3)
            _put(image, 'LEFT', (lx + 8, max(ly - 12, 20)), 0.45, (255, 0, 255), 2)

        # ── 우측 경계: 마찬가지로 세로선 ──
        if right is not None:
            rx, ry = right
            cv2.line(image, (rx, 0), (rx, h), (255, 0, 255), 2)
            cv2.circle(image, (rx, ry), 11, (255, 0, 255), 3)
            _put(image, 'RIGHT', (rx + 8, max(ry - 12, 20)), 0.45, (255, 0, 255), 2)

        # ── 깊이 기준선(중간 마커): 가로 주황색 선 ──
        if mid is not None:
            mx, my = mid
            cv2.line(image, (0, my), (w, my), (0, 180, 255), 2)
            cv2.circle(image, (mx, my), 11, (0, 180, 255), 3)
            _put(image, 'MID / DEPTH', (mx + 8, max(my - 12, 20)), 0.45, (0, 180, 255), 2)

        # ── 세 마커 모두 있으면 침대 후보 영역을 반투명 보라색으로 표시 ──
        if left is not None and right is not None and mid is not None:
            lx, _ = left
            rx, _ = right
            _, my = mid

            # 좌/우 마커 x좌표를 양 끝으로, 중간 마커 y좌표를 아래쪽 경계로 하는 사각형
            x_min = min(lx, rx)
            x_max = max(lx, rx)

            # 사각형의 네 꼭짓점 (시계방향)
            pts = np.array([
                [x_min, 0],     # 좌상
                [x_max, 0],     # 우상
                [x_max, my],    # 우하
                [x_min, my],    # 좌하
            ], dtype=np.int32)

            # 반투명 fill을 위해 "오버레이" 이미지를 따로 만들어 가중 평균하는 트릭
            #   원본:오버레이 = 0.88:0.12 → 살짝만 보라색 입혀짐
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], (255, 0, 255))
            cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)
            # 그 위에 외곽선을 또렷하게 한 번 더
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

        # ── BED CHECK 결과 텍스트: 결과별로 색상 다름 ──
        if bed_decision == 'IN_BED':
            color = (0, 255, 0)         # 초록 = OK
            text = 'BED CHECK: IN_BED'
        elif bed_decision == 'OUT_OF_BED':
            color = (0, 0, 255)         # 빨강 = 위험
            text = 'BED CHECK: OUT_OF_BED'
        elif bed_decision == 'NO_MARKER':
            color = (0, 165, 255)       # 주황 = 마커 못 찾음
            text = 'BED CHECK: NO_MARKER'
        else:
            color = (200, 200, 200)     # 회색 = 그 외
            text = f'BED CHECK: {bed_decision}'

        _put(image, f'MARKERS: {len(bed_markers)}', (20, 240), 0.50, (0, 255, 255), 2)
        _put(image, text, (20, 270), 0.55, color, 2)

    # ─────────────────────────────────────────────────────────────────
    # 좌상단 텍스트 오버레이 + 화면 중앙/가장자리 가이드 라인 그리기
    #
    #   여기서 그리는 텍스트들은 amr_detect 노드가 매 프레임 채워 보내는 상태값:
    #     status   : 'PERSON: YES'/'PERSON: NO' 등 ('YES' 포함되면 초록, 아니면 빨강)
    #     mode     : 현재 모드 (예: 'MODE: SEARCH_ROTATE')
    #     align    : 정렬/거리 등 디테일 정보
    #     pose_text: 'POSE: Lying' 같은 자세 표시
    #     processed_count: 지금까지 처리한 사람 수
    #
    #   가이드 라인:
    #     - 화면 중앙 세로선 (시안)
    #     - 좌/우 edge_margin_px 위치 세로선 (주황) → 사람 끝에 붙었는지 판단 시 도움
    # ─────────────────────────────────────────────────────────────────
    def _draw_overlay(
        self,
        frame,
        annotated,
        status,
        mode,
        align,
        pose_text,
        pose_label,
        start_yaw,
        current_yaw,
        processed_count,
    ):
        # status 에 'YES' 가 들어있으면 초록, 그 외엔 빨강
        status_color = (0, 255, 0) if 'YES' in status else (0, 0, 255)

        # (텍스트, 위치, 크기, 색)을 한 데 모아서 루프로 그리면 코드가 깔끔
        overlay_lines = [
            (status,    (20, 30),  0.45, status_color),
            (mode,      (20, 65),  0.45, (255, 255, 0)),     # 모드 = 시안
            (align,     (20, 100), 0.35, (255, 255, 255)),   # 정렬 = 흰색
            (pose_text, (20, 135), 0.40, POSE_COLOR.get(pose_label, _GRAY)),  # 자세별 색
        ]
        for text, pos, scale, color in overlay_lines:
            _put(annotated, text, pos, scale, color)

        # ── yaw 정보가 둘 다 들어와 있을 때만 표시 (시작 방향 vs 현재 방향) ──
        if start_yaw is not None and current_yaw is not None:
            _put(
                annotated,
                f'START_YAW: {math.degrees(start_yaw):.1f}  CUR_YAW: {math.degrees(current_yaw):.1f}',
                (20, 170), 0.49, (200, 255, 200),
            )

        # 처리된 사람 수
        _put(annotated, f'PROCESSED: {processed_count}',
             (20, 205), 0.49, (180, 220, 255))

        # ── 가이드 라인들 ──
        # 주의: frame 의 shape를 쓰는 건 "원본 크기" 기준이기 때문 (annotated와 같음)
        h, w = frame.shape[:2]
        cx = w // 2  # 화면 가로 중앙

        # (x좌표, 색, 두께) 세 가지 세로선:
        #   1) 화면 중앙선 - 시안 (사람을 가운데 정렬할 때 기준)
        #   2) 왼쪽 마진선 - 주황
        #   3) 오른쪽 마진선 - 주황
        for x, color, thickness in [
            (cx,                         (0, 255, 255), 2),
            (self.edge_margin_px,        (255, 100, 0), 1),
            (w - self.edge_margin_px,    (255, 100, 0), 1),
        ]:
            cv2.line(annotated, (x, 0), (x, h), color, thickness)

    # ─────────────────────────────────────────────────────────────────
    # 외부에서 호출하는 메인 진입점
    #   - 전달받은 정보들로 모든 시각화를 합성한 뒤 OpenCV 창에 띄움
    #   - cv2.waitKey로 키 입력 1ms 대기 → ESC(27) 눌렸으면 True 반환
    #     (호출자는 True가 오면 종료 처리를 시작)
    # ─────────────────────────────────────────────────────────────────
    def render(
        self,
        frame,
        people=None,
        target_index=None,
        status='PERSON: NO',
        mode='MODE: WAITING',
        align='',
        pose_text='POSE: ---',
        current_pose_label='Unknown',
        start_yaw=None,
        current_yaw=None,
        processed_count=0,
        bed_markers=None,
        bed_left_marker=None,
        bed_right_marker=None,
        bed_mid_marker=None,
        bed_decision='UNKNOWN',
    ):
        # 원본 frame 을 직접 그리면 "다른 곳에서도 이 frame을 쓰는 경우" 영향을 줌
        # → copy() 해서 annotated에 그림 (원본 보존)
        annotated = frame.copy()

        # 1) 침대 마커 관련 시각화 (있으면 표시, 없으면 자동 스킵)
        self._draw_bed_marker_info(
            annotated,
            bed_markers=bed_markers,
            bed_left_marker=bed_left_marker,
            bed_right_marker=bed_right_marker,
            bed_mid_marker=bed_mid_marker,
            bed_decision=bed_decision,
        )

        # 2) 사람 박스/스켈레톤/라벨
        self._draw_people(annotated, people or [], target_index)

        # 3) 좌상단 텍스트 + 가이드 선
        self._draw_overlay(
            frame, annotated,
            status, mode, align, pose_text,
            current_pose_label,
            start_yaw, current_yaw, processed_count,
        )

        # ── 디스플레이용 확대 ──
        # 카메라 원본 해상도가 작을 수 있으므로 display_scale 만큼 키워서 띄움
        h, w = annotated.shape[:2]
        display = cv2.resize(
            annotated,
            (int(w * self.display_scale), int(h * self.display_scale)),
        )

        cv2.imshow(self.window_name, display)

        # waitKey(1) : 1ms 동안 키 입력 받기 (창 갱신을 위해 필수 호출)
        # & 0xFF    : 일부 환경에서 상위 비트가 섞여 들어와서 마스크 처리 (관용구)
        # 27        : ESC 키의 ASCII 코드
        return (cv2.waitKey(1) & 0xFF) == 27

    # 모든 OpenCV 창 닫기 (노드 종료 시 호출)
    def close(self):
        cv2.destroyAllWindows()
