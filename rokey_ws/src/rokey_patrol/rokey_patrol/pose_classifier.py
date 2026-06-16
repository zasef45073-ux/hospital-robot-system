#!/usr/bin/env python3
"""
================================================================================
  자세 분류(Pose Classification) 모듈
================================================================================

[전체 개요]
사람의 키포인트(코, 어깨, 무릎 등 17개 관절 좌표)를 보고
"Standing(서있음) / Sitting(앉음) / Lying(누움) / Unknown(모름)" 중 하나로 분류합니다.

이 파일은 사실 3가지 역할을 한 파일에 담고 있습니다:

  1) classify_pose() - 핵심 알고리즘 함수
        키포인트와 박스를 입력받아 자세 라벨을 반환.
        amr_detect.py 같은 다른 노드에서도 이 함수만 가져다 씀.

  2) YOLOPoseProcessor + main() - 단독 실행 모드
        `python pose_classifier.py` 로 직접 돌리면 웹캠을 켜고
        실시간으로 사람을 보면서 결과를 영상/JSON/CSV로 저장하는 데모/디버깅용.

  3) PoseClassifier 클래스 (파일 맨 아래)
        amr_detect.py 가 사용하는 객체 인터페이스. 내부적으로 classify_pose() 호출.

[자세 판정 알고리즘 핵심]
"여러 신호를 점수로 환산해서 합산"하는 score voting 방식입니다.
   - lying_score(누움 가능성)와 vertical_score(서/앉음 가능성)에 점수를 누적
   - 한 가지 단서만으로 결론 내지 않음 → 오탐 방지
   - 사용하는 단서:
       (A) 어깨-엉덩이 연결선의 기울기 (수평이면 누움 가능성↑)
       (B) 박스의 가로/세로 비율 (가로로 길면 누움 쪽)
       (C) 머리-골반 화면상 높이차 (작으면 누움 쪽)
       (D) 키포인트가 한 줄로 늘어선 정도 (PCA로 측정)
       (E) 어깨→엉덩이→무릎→발목이 위→아래 순서로 잘 정렬되어 있는지
   - lying_score 가 충분히 크고 vertical_score 보다 분명히 높을 때만 Lying 확정
   - vertical 계열이면 무릎 각도/골반 위치로 Standing/Sitting 세부 판정

[COCO 키포인트 인덱스 참고]
  0:nose / 1,2:eye / 3,4:ear / 5,6:shoulder / 7,8:elbow / 9,10:wrist
  11,12:hip / 13,14:knee / 15,16:ankle  (왼쪽이 먼저 오는 인덱스)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 표준 라이브러리
# ─────────────────────────────────────────────────────────────────────────────
import json
import csv
import time
import os
import shutil
import sys
from collections import deque  # 메모리 폭증 방지를 위해 최근 N개만 메모리에 유지
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# 외부 라이브러리
# ─────────────────────────────────────────────────────────────────────────────
import cv2                       # OpenCV: 웹캠 캡쳐 / 화면 표시 / 이미지 저장
import numpy as np               # 좌표·벡터 연산
from ultralytics import YOLO     # YOLOv8 모델 (단독 실행 모드에서 사용)


# -----------------------------------------------------------------------------
# 전역 설정값(임계값)
#   - 모두 자세 판정에 쓰이는 튜닝 파라미터
#   - main() 안에서 argparse / config json 으로 덮어쓸 수 있음(전역 변수임에 주의)
# -----------------------------------------------------------------------------
KPT_CONF_TH = 0.45              # 키포인트 신뢰도 임계값 (이 미만이면 그 점은 안 쓴다)

STANDING_KNEE_ANGLE_TH = 150.0  # 무릎 각도가 이 이상(거의 펴짐)이면 Standing 후보
SITTING_KNEE_ANGLE_TH  = 140.0  # 무릎 각도가 이 이하(굽힘)이면 Sitting 후보

# Lying 보조 기준
LYING_BOX_ASPECT_TH  = 1.35     # box_w / box_h 가 이보다 크면 누워있을 가능성↑ (가로로 누운 모양)
TORSO_HORIZONTAL_TH  = 26.0     # 어깨-엉덩이 연결선 기울기가 26도 이하면 거의 수평 → 누움 쪽
TORSO_VERTICAL_TH    = 28.0     # 28도 이상이면 세워져 있는 쪽
                                # (26~28 사이는 애매구간으로 두고 점수 가산 안 함)

# 점수 누적 방식 임계값
LYING_SCORE_TH    = 4.0         # lying_score 가 이 이상이고 vertical_score 보다 충분히 크면 Lying
VERTICAL_SCORE_TH = 3.0         # vertical_score 가 이 이상일 때 Standing/Sitting 세부 판별로 진입

# Lying 보조 특징 임계값
HEAD_HIP_Y_RATIO_TH  = 0.18     # |머리.y - 골반.y| / 박스높이 가 이 이하면 거의 한 평면 → 누움 쪽
LINEARITY_RATIO_TH   = 0.25     # PCA 2번째/1번째 고유값 비율, 작을수록 키포인트가 한 줄에 가까움(누움)

# Standing 세부 판별 보조
STANDING_HIP_ABOVE_KNEE_RATIO_TH = 0.08   # 골반이 무릎보다 박스높이의 8% 이상 위에 있어야 Standing 인정

FRAME_SAVE_INTERVAL_SEC = 1.0   # 단독 실행 모드에서 프레임 저장 최소 간격(초)
IN_MEMORY_EVENT_LIMIT   = 20    # 메모리에 보관할 최근 이벤트 개수 (오래된 건 자동으로 빠짐)


# -----------------------------------------------------------------------------
# 각도/포인트 계산용 헬퍼
# -----------------------------------------------------------------------------

def _angle_3pts(a, b, c):
    """
    세 점 a, b, c 가 만드는 ∠ABC 각도(b가 꼭짓점)를 도(degree)로 반환.

    예: 엉덩이-무릎-발목 → 무릎이 얼마나 굽혀져 있는지 측정할 때 씀
        (펴져 있으면 180도에 가깝고, 굽으면 작아짐)

    반환:
        float : 0~180 도
        None  : 두 벡터 중 길이가 거의 0이라 각도 정의 불가
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # b 기준의 두 벡터를 만든 뒤 코사인 법칙 사용
    ba = a - b
    bc = c - b

    # 영벡터에 가까우면 각도가 정의되지 않음 (NaN/예외 방지)
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return None

    # 코사인 = (ba·bc) / (|ba| |bc|)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # 부동소수 오차로 인한 -1~1 범위 이탈 방지 (clip 안 하면 arccos가 NaN 낼 수 있음)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _line_angle_deg(p1, p2):
    """
    두 점을 잇는 선분이 수평선과 이루는 각도(0~90도)를 반환.

    - 0에 가까움  : 거의 수평 (예: 누워있는 사람의 어깨-엉덩이 라인)
    - 90에 가까움 : 거의 수직 (서/앉아있는 사람의 어깨-엉덩이 라인)

    반환:
        float : 0~90
        None  : 두 점이 사실상 같아서 방향이 정의되지 않음
    """
    x1, y1 = p1
    x2, y2 = p2
    # 절댓값을 쓰는 이유: 선분이 좌상↔우하든 우상↔좌하든 "기울기 크기"만 보면 충분
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx < 1e-6 and dy < 1e-6:
        return None

    # atan2(dy, dx): dx 축(수평)을 기준으로 한 각도 (90도 = 수직)
    return float(np.degrees(np.arctan2(dy, dx)))


def _get_point(kpt, idx):
    """
    kpt[idx] 키포인트의 confidence가 임계값 이상이면 (x, y, conf) 튜플 반환,
    아니면 None (= 못 믿겠으니 쓰지 마라).

    kpt 형태는 (17, 3) 가정: 각 행이 (x, y, conf).
    """
    x, y, c = kpt[idx]
    if c >= KPT_CONF_TH:
        return (float(x), float(y), float(c))
    return None


def _mean_point(kpt, indices):
    """
    여러 키포인트의 평균 좌표를 반환 (예: 좌/우 어깨의 가운데 = 어깨 중심).

    - indices 중에서 confidence를 통과한 것들만 가지고 평균
    - 하나도 안 남으면 None 반환
    """
    pts = [_get_point(kpt, i) for i in indices]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    return (
        float(np.mean([p[0] for p in pts])),
        float(np.mean([p[1] for p in pts])),
    )


def _keypoint_linearity_ratio(kpt, box_w, box_h):
    """
    키포인트들이 한 직선 위에 가깝게 늘어서 있는지를 PCA로 측정.

    원리:
        키포인트 좌표들의 공분산 행렬 고유값을 보면
        - 점들이 한 줄로 길게 퍼져있으면  → 큰 고유값 1개 + 작은 고유값 1개
        - 점들이 둥글게 퍼져있으면         → 비슷한 두 고유값
        그래서 (작은 고유값 / 큰 고유값) 비율이 작을수록 "한 줄에 가깝다"는 뜻.

    반환:
        float : 0에 가까울수록 선형 (Lying 가능성↑)
        None  : 유효 키포인트가 5개 미만이거나 수치적으로 계산 실패
    """
    pts = []
    for i in range(len(kpt)):
        p = _get_point(kpt, i)
        if p is not None:
            pts.append([p[0], p[1]])

    # 점이 너무 적으면 PCA 의미가 없어서 포기
    if len(pts) < 5:
        return None

    pts = np.asarray(pts, dtype=np.float32)

    # bbox 크기로 정규화 → 카메라 거리(피사체가 크게/작게 잡힘)에 영향 받지 않게 함
    pts[:, 0] /= max(box_w, 1e-6)
    pts[:, 1] /= max(box_h, 1e-6)

    mean     = np.mean(pts, axis=0, keepdims=True)
    centered = pts - mean
    cov      = np.cov(centered.T)

    try:
        # eigvalsh: 대칭행렬용. 정렬해서 큰 → 작은 순서로 만듦
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    except np.linalg.LinAlgError:
        return None

    # 너무 작으면 0으로 나누기 위험 → 포기
    if len(eigvals) < 2 or eigvals[0] < 1e-8:
        return None

    return float(eigvals[1] / eigvals[0])


def _vertical_stack_score(sh, hp, kn, an, box_h):
    """
    어깨(sh) → 엉덩이(hp) → 무릎(kn) → 발목(an) 의 y좌표가
    "위에서 아래로" 차례로 내려가는지를 점수화.

    영상 좌표계에서는 화면 위쪽이 y=0, 아래쪽이 y=큰값 이라는 점 주의.
    따라서 y가 점점 커지는 게 "위→아래" 순서임.

    반환:
        0.0 ~ 1.7 사이의 점수
        - 모든 인접 구간이 위→아래 순서이면 +1.0
        - y가 박스 높이의 35% 이상으로 펼쳐져 있으면 +0.7  (충분히 키 큰 자세)
    """
    # None 인 점은 제외하고 사용 가능한 것만 모음
    pts = [p for p in [sh, hp, kn, an] if p is not None]
    if len(pts) < 3:
        return 0.0

    ys         = [p[1] for p in pts]
    total      = len(ys) - 1
    # 인접한 두 점 비교 (i번째 y가 i+1번째 y보다 작거나 같으면 = 위에 있으면) 카운트
    increasing = sum(1 for i in range(total) if ys[i] <= ys[i + 1])

    score = 0.0

    # (1) 모든 인접 구간이 정상 순서면 보너스
    if total > 0 and increasing == total:
        score += 1.0

    # (2) 위쪽~아래쪽 키포인트가 박스 세로의 35% 이상 차지하면 또 보너스
    #     너무 짧으면 그냥 카메라 가까이서 부분만 잡힌 것일 수 있음
    y_span = (max(ys) - min(ys)) / max(box_h, 1e-6)
    if y_span >= 0.35:
        score += 0.7

    return score


# -----------------------------------------------------------------------------
# 자세 분류 핵심 함수
# -----------------------------------------------------------------------------
def classify_pose(kpt, box):
    """
    점수 누적(score voting) 방식 자세 판정 함수.

    여러 단서를 각각 점수로 환산해 lying_score / vertical_score 에 더해넣고,
    종합 점수를 비교해 최종 라벨을 결정합니다. 단일 단서로 결론을 내지 않기 때문에
    오탐(예: 아이를 안고 서있는데 박스가 가로로 길어서 누움으로 오인)에 강합니다.

    입력:
        kpt: (17, 3) numpy array. 각 행 = (x, y, conf). COCO 키포인트 표준.
        box: [x1, y1, x2, y2] - 사람을 둘러싼 박스의 좌상/우하 좌표

    출력:
        "Standing" / "Sitting" / "Lying" / "Unknown" 중 하나
    """
    x1, y1, x2, y2 = box
    box_w = float(x2 - x1)
    box_h = float(y2 - y1)

    # 박스가 비정상이면 분류 불가
    if box_w <= 1.0 or box_h <= 1.0:
        return "Unknown"

    # ---------- 대표 관절 좌표(중심점) 계산 ----------
    # 좌/우가 둘 다 있는 부위는 평균을 내서 "중심점"으로 사용
    # COCO 키포인트 인덱스 참고:
    #   머리 : 0(코), 1,2(눈), 3,4(귀)
    #   어깨 : 5, 6
    #   골반 : 11, 12
    #   무릎 : 13, 14
    #   발목 : 15, 16
    sh = _mean_point(kpt, [5, 6])           # 어깨 중심
    hp = _mean_point(kpt, [11, 12])         # 골반 중심
    hd = _mean_point(kpt, [0, 1, 2, 3, 4])  # 머리 중심 (얼굴 부위 평균)
    kn = _mean_point(kpt, [13, 14])         # 무릎 중심
    an = _mean_point(kpt, [15, 16])         # 발목 중심

    # ---------- 주요 특징 계산 ----------
    # (1) 몸통(어깨-엉덩이)의 기울기. 0=수평, 90=수직
    torso_angle      = _line_angle_deg(sh, hp) if (sh and hp) else None

    # (2) 박스의 가로:세로 비율
    aspect           = box_w / box_h

    # (3) 키포인트가 한 줄로 모여있는 정도(PCA)
    linearity_ratio  = _keypoint_linearity_ratio(kpt, box_w, box_h)

    # (4) 머리와 골반의 y(높이) 차이를 박스 높이로 정규화
    #     누워있으면 머리와 골반이 거의 같은 y라 이 값이 작아짐
    head_hip_y_ratio = (
        abs(hp[1] - hd[1]) / max(box_h, 1e-6)
        if (hd and hp) else None
    )

    # (5) 무릎 각도 - 좌/우 다리 중 더 신뢰도 높은 쪽 선택
    #     엉덩이-무릎-발목 세 점이 모두 보일 때만 계산 가능
    candidates = []
    for hip_i, knee_i, ankle_i in [(11, 13, 15), (12, 14, 16)]:  # (왼다리, 오른다리)
        h_pt = _get_point(kpt, hip_i)
        k_pt = _get_point(kpt, knee_i)
        a_pt = _get_point(kpt, ankle_i)
        if h_pt and k_pt and a_pt:
            ang = _angle_3pts(h_pt[:2], k_pt[:2], a_pt[:2])
            if ang is not None:
                conf_avg = (h_pt[2] + k_pt[2] + a_pt[2]) / 3.0
                candidates.append((ang, conf_avg))

    # 후보 중 평균 신뢰도가 가장 높은 무릎 각도 채택. 없으면 None
    knee_angle = max(candidates, key=lambda x: x[1])[0] if candidates else None

    # ---------- 점수 누적 ----------
    lying_score    = 0.0
    vertical_score = 0.0

    # (A) 몸통축 각도 - 가장 강력한 단서이므로 가중치 3.0
    #     수평이면 lying, 수직이면 vertical, 그 사이는 애매구간(가산 안 함)
    if torso_angle is not None:
        if torso_angle <= TORSO_HORIZONTAL_TH:
            lying_score    += 3.0
        elif torso_angle >= TORSO_VERTICAL_TH:
            vertical_score += 3.0

    # (B) bbox 가로/세로 비율 - 보조 단서 (단독으로는 절대 결론 못 냄)
    #     가로로 길면 누움 쪽, 세로로 길면 서/앉음 쪽
    if aspect >= LYING_BOX_ASPECT_TH:
        lying_score    += 1.0
    elif aspect <= 0.85:
        vertical_score += 0.8

    # (C) 머리-골반 높이차
    #     서/앉아있으면 머리는 위, 골반은 아래라 차이가 어느 정도 큼
    if head_hip_y_ratio is not None:
        if head_hip_y_ratio <= HEAD_HIP_Y_RATIO_TH:
            lying_score    += 1.5
        else:
            vertical_score += 1.0

    # (D) 키포인트 선형성 - 한 줄에 가까울수록 누움 쪽
    if linearity_ratio is not None:
        if linearity_ratio <= LINEARITY_RATIO_TH:
            lying_score    += 1.2
        else:
            vertical_score += 0.6

    # (E) 어깨→엉덩이→무릎→발목 세로 정렬 점수 (vertical 계열 보강)
    vertical_score += _vertical_stack_score(sh, hp, kn, an, box_h)

    # ---------- 1차 분류: Lying 인지 결정 ----------
    # "lying_score 가 임계값 이상" + "vertical_score 보다 1점 이상 큼"
    # 두 조건을 동시에 만족할 때만 Lying 으로 확정 → 단일 단서 오탐 방지
    if lying_score >= LYING_SCORE_TH and lying_score >= vertical_score + 1.0:
        return "Lying"

    # ---------- 2차 분류: Standing / Sitting 세부 판별 ----------
    if vertical_score >= VERTICAL_SCORE_TH:
        # ── Standing 조건: 무릎이 충분히 펴졌고 + 골반이 무릎보다 충분히 위에 ──
        if knee_angle is not None and hp is not None and kn is not None:
            hip_knee_dy = (kn[1] - hp[1]) / max(box_h, 1e-6)  # 양수 = 골반이 무릎보다 위
            print(f"vertical_score={vertical_score:.2f}, knee_angle={knee_angle:.1f}, hip_knee_dy={hip_knee_dy:.3f}")
            if (
                knee_angle >= STANDING_KNEE_ANGLE_TH
                and hip_knee_dy >= STANDING_HIP_ABOVE_KNEE_RATIO_TH
            ):
                return "Standing"

        # ── Sitting 조건: 무릎이 굽혀져 있음 ──
        if knee_angle is not None and knee_angle <= SITTING_KNEE_ANGLE_TH:
            return "Sitting"

        # ── 무릎 정보가 부족할 때 보조 판단:
        #     상체는 어느 정도 세워져 있는데(머리가 골반보다 위에 충분히)
        #     무릎-발목 거리가 짧으면 → 다리가 가려진 앉은 자세일 가능성
        if hd is not None and hp is not None:
            head_hip_dy = (hp[1] - hd[1]) / max(box_h, 1e-6)
            if head_hip_dy >= 0.22 and kn is not None and an is not None:
                knee_ankle_dy = abs(an[1] - kn[1]) / max(box_h, 1e-6)
                if knee_ankle_dy < 0.18:
                    return "Sitting"

    # 어떤 조건에도 확실히 들어맞지 않으면 Unknown
    # (예: 비스듬히 기울어진 자세, 걷는 중간 자세, 반쯤 앉은 자세 등)
    return "Unknown"


# 자세별 색상 (단독 실행 모드에서 박스 그릴 때 사용)
POSE_COLOR = {
    "Standing": (0, 255, 0),     # 초록 - BGR 순서임에 주의 (OpenCV 관행)
    "Sitting":  (0, 165, 255),   # 주황
    "Lying":    (0, 0, 255),     # 빨강
    "Unknown":  (200, 200, 200), # 회색
}


# =============================================================================
# 아래부터는 "단독 실행 모드"용 코드 (python pose_classifier.py 실행 시 사용)
#   다른 노드에서 import 해 쓸 때는 이 클래스/main 은 건드리지 않음
# =============================================================================

class YOLOPoseProcessor:
    """
    웹캠 → YOLO 추론 → 자세 분류 → 시각화/저장 까지 처리하는 데모용 프로세서.

    동작 흐름:
        1) 웹캠을 열고
        2) grab_fps 주기로 프레임을 가져오고
        3) process_fps 주기로만 YOLO 추론 (CPU 절약)
        4) 결과(자세 라벨, 키포인트, 박스 등)를 JSONL로 한 줄씩 추가 저장
        5) 'q' 키 누르면 종료
    """

    def __init__(self, model, output_dir, cam_index=1, grab_fps=30.0, process_fps=30.0):
        self.model      = model
        self.output_dir = output_dir

        self.cam_index   = cam_index
        # 0 이하면 "제한 없음"으로 처리하기 위해 0.0으로 통일
        self.grab_fps    = float(grab_fps)    if grab_fps    and grab_fps    > 0 else 0.0
        self.process_fps = float(process_fps) if process_fps and process_fps > 0 else 0.0

        # 메모리 사용을 막기 위해 deque(maxlen=N): 가득 차면 오래된 게 자동 제거됨
        self.csv_output  = deque(maxlen=IN_MEMORY_EVENT_LIMIT)
        self.confidences = deque(maxlen=IN_MEMORY_EVENT_LIMIT)

        self.max_person_count = 0      # 한 프레임에 보였던 최대 인원 수 (통계용)
        self.should_shutdown  = False  # 'q' 키 등으로 종료 요청됐는지

        # 평균 신뢰도 계산용 누적값 (리스트로 다 쌓으면 메모리 큼 → 합/개수만 보관)
        self._conf_sum   = 0.0
        self._conf_count = 0

        # 마지막으로 프레임 이미지를 저장한 시각 (저장 간격 제어용)
        self._last_frame_save_ts = 0.0

        # JSONL: 한 줄에 JSON 객체 하나씩 적는 포맷. append 친화적이라 스트리밍 저장에 좋음
        self._events_jsonl_path = os.path.join(self.output_dir, "pose_events.jsonl")

    def run(self):
        """
        메인 루프. 웹캠 열고 → 프레임 받으며 추론/시각화/저장 → 'q' 누르면 종료.
        """
        # CAP_DSHOW: Windows에서 DirectShow 백엔드 사용 (더 빠르게 캠 열림)
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

        # 카메라 자체의 FPS도 맞춰주기 (지원하는 카메라에 한해)
        if self.grab_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.grab_fps)

        print("Streaming 시작... 'q'를 누르면 종료됩니다.")
        print(f"- grab_fps(가져오기): {self.grab_fps}")
        print(f"- process_fps(추론): {self.process_fps}")

        latest_frame    = None  # 가장 최근에 캠에서 받아온 raw 프레임
        annotated_frame = None  # YOLO 결과 그려진 프레임 (이전 추론 결과 그대로 보여줌)

        # 단순 스케줄러: 시각이 next_*_ts 를 지났을 때만 grab/proc 실행
        now          = time.time()
        next_grab_ts = now
        next_proc_ts = now

        # FPS → 주기(초)
        grab_interval = (1.0 / self.grab_fps)    if self.grab_fps    > 0 else 0.0
        proc_interval = (1.0 / self.process_fps) if self.process_fps > 0 else 0.0

        # JSONL 파일을 append 모드로 열어 둠 (루프 도는 동안 계속 추가)
        with open(self._events_jsonl_path, "a", encoding="utf-8") as jf:
            while not self.should_shutdown:
                now = time.time()

                # ── 1) 프레임 가져오기 (grab_fps 제한 적용) ──
                if (self.grab_fps <= 0) or (now >= next_grab_ts):
                    ret, img     = cap.read()
                    next_grab_ts = now + grab_interval
                    if ret:
                        latest_frame = img
                    else:
                        # 프레임 못 받았으면 잠시 쉬고 재시도
                        time.sleep(0.01)
                        continue

                # ── 2) 추론 수행 (process_fps 제한 적용) ──
                do_proc = (latest_frame is not None) and (
                    (self.process_fps <= 0) or (now >= next_proc_ts)
                )
                if do_proc:
                    annotated_frame, person_count = self._process_frame(latest_frame, jf)
                    if self.process_fps > 0:
                        next_proc_ts = now + proc_interval

                # ── 3) 화면 출력 (annotated 가 있으면 그것을, 없으면 raw 그대로) ──
                show_img = annotated_frame if annotated_frame is not None else latest_frame
                if show_img is not None:
                    cv2.imshow("YOLOv8 Pose Detection", show_img)

                # 종료 키 처리: 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("종료 중...")
                    self.should_shutdown = True

        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, img, jsonl_fp):
        """
        프레임 한 장을 받아 YOLO 추론 + 시각화 + 사람별 처리(_process_person)까지 수행.

        반환:
            annotated_frame : 박스/라벨이 그려진 이미지
            person_count    : 이 프레임에서 검출된 사람 수
        """
        # stream=True : generator로 받음 (메모리 절약 & 한 장씩 처리)
        results = self.model(img, stream=True)

        person_count    = 0
        annotated_frame = img.copy()

        for r in results:
            # YOLO 기본 시각화(스켈레톤 + 박스) 한 번 그려두고,
            annotated_frame = r.plot()

            # 키포인트나 박스가 없으면 사람별 후속 처리 불가 → 스킵
            if r.keypoints is None or r.boxes is None:
                continue

            person_count = len(r.keypoints.data)
            self.max_person_count = max(self.max_person_count, person_count)

            # 사람 한 명씩 자세 분류 + 결과 누적
            for i, kpt in enumerate(r.keypoints.data):
                now = time.time()
                self._process_person(
                    i=i,
                    r=r,
                    kpt=kpt,
                    annotated_frame=annotated_frame,
                    jsonl_fp=jsonl_fp,
                    current_time=now,
                )

                # 사람이 있을 때만 일정 간격으로 프레임 이미지 저장
                if person_count > 0:
                    self._save_frame(annotated_frame, current_time=now)

        # 프레임 좌상단에 인원 수 표시
        cv2.putText(
            annotated_frame,
            f"Persons: {person_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return annotated_frame, person_count

    def _process_person(self, i, r, kpt, annotated_frame, jsonl_fp, current_time):
        """
        검출된 사람 한 명에 대한 처리.
            - 신뢰도 통계 누적
            - 자세 분류
            - 박스 + 라벨 시각화 (자세별 색상)
            - 이벤트를 메모리(deque) + JSONL 파일에 동시 저장
        """
        # 사람 박스의 신뢰도 (스켈레톤 키포인트 conf 와는 별개)
        conf = r.boxes.conf[i].item()

        # 평균 신뢰도 계산용 누적 (장시간 실행 시 메모리 누수 방지: 합과 개수만 보관)
        self._conf_sum   += float(conf)
        self._conf_count += 1
        self.confidences.append(conf)

        # 키포인트 텐서 → 파이썬 list (JSON 직렬화 가능 형태로)
        kpts_list = kpt.cpu().numpy().tolist()
        # 박스 좌표
        box_xyxy = r.boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]

        # 자세 분류 (이 모듈의 핵심 함수 호출)
        pose_label = classify_pose(kpt.cpu().numpy(), box_xyxy)

        # 시각화: 박스 + 자세 라벨 (색상은 POSE_COLOR 사전 사용)
        x1, y1, x2, y2 = box_xyxy.astype(int)
        color = POSE_COLOR.get(pose_label, (200, 200, 200))

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            pose_label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # 이벤트 한 건의 데이터 묶음
        event = {
            "ts":           current_time,                                  # 발생 시각(Unix timestamp)
            "person_index": i,                                             # 프레임 내 순번 (트래킹 ID 아님)
            "conf":         float(conf),                                   # 박스 신뢰도
            "kpts":         kpts_list,                                     # 17 keypoints (x, y, conf)
            "pose_label":   pose_label,                                    # 자세 분류 결과
            "box_xyxy":     [float(x1), float(y1), float(x2), float(y2)],  # 박스
        }

        # 메모리(최근 N개만)
        self.csv_output.append([current_time, i, float(conf), kpts_list, pose_label])
        # 파일(JSONL) - 한 줄에 한 객체
        jsonl_fp.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_frame(self, annotated_frame, current_time):
        """
        FRAME_SAVE_INTERVAL_SEC 마다 최대 1장만 이미지로 저장.
        (매 프레임 저장하면 디스크 폭발하므로 throttle)
        """
        now = time.time()
        if (now - self._last_frame_save_ts) < FRAME_SAVE_INTERVAL_SEC:
            return

        self._last_frame_save_ts = now
        filename = f"pose_{current_time}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, filename), annotated_frame)

    def save_output(self):
        """
        프로그램 종료 시 결과 정리 저장.
            - pose_data.json : 메모리에 남은 최근 N개 이벤트 (요약 보기 좋음)
            - pose_events.jsonl : 실행 중 누적된 전체 이벤트 (스트리밍 저장된 파일)
            - statistics.csv : 최대 인원 수 + 평균 confidence
        """
        # 1) 최근 이벤트만 들어있는 JSON
        pose_data_path = os.path.join(self.output_dir, "pose_data.json")
        with open(pose_data_path, "w", encoding="utf-8") as f:
            json.dump(list(self.csv_output), f, ensure_ascii=False)

        # 2) 통계 CSV
        stats_path = os.path.join(self.output_dir, "statistics.csv")
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Max Persons Detected", "Average Confidence"])

            avg_conf = (self._conf_sum / self._conf_count) if self._conf_count > 0 else 0.0
            writer.writerow([self.max_person_count, avg_conf])

        print(f"결과가 {self.output_dir} 폴더에 저장되었습니다.")
        print(f"- 최근 이벤트 JSON: {pose_data_path}")
        print(f"- 전체 이벤트 JSONL: {self._events_jsonl_path}")
        print(f"- 통계 CSV: {stats_path}")


# -----------------------------------------------------------------------------
# 설정 로딩 헬퍼들 (단독 실행 모드 전용)
# -----------------------------------------------------------------------------
def load_config_json(path):
    """JSON 설정 파일을 dict로 로드. 파일이 없거나 깨졌으면 빈 dict 반환."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Config 파일을 찾을 수 없습니다: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Config JSON 로드 실패: {e}")
        return {}


def pick(cfg, name, argval, default=None):
    """
    설정값 우선순위 결정 헬퍼.
    우선순위:  (1) 커맨드라인 인자(argval)  >  (2) config json(cfg[name])  >  (3) default
    None 이면 "값이 안 들어왔다"는 의미로 사용.
    """
    if argval is not None:
        return argval
    if name in cfg and cfg[name] is not None:
        return cfg[name]
    return default


# -----------------------------------------------------------------------------
# 단독 실행 진입점
#   `python pose_classifier.py [--옵션...]` 으로 실행될 때만 동작
# -----------------------------------------------------------------------------
def main():
    # 모듈 상단 전역 임계값을 main() 안에서 덮어쓰기 위해 global 선언
    # (다른 모듈이 import 해 쓸 때는 이 main() 자체가 호출되지 않으므로 영향 없음)
    global KPT_CONF_TH
    global STANDING_KNEE_ANGLE_TH, SITTING_KNEE_ANGLE_TH
    global LYING_BOX_ASPECT_TH, TORSO_HORIZONTAL_TH, TORSO_VERTICAL_TH
    global LYING_SCORE_TH, VERTICAL_SCORE_TH
    global HEAD_HIP_Y_RATIO_TH, LINEARITY_RATIO_TH
    global STANDING_HIP_ABOVE_KNEE_RATIO_TH
    global FRAME_SAVE_INTERVAL_SEC

    # ── 커맨드라인 인자 정의 ──
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="pose_args.json", help="설정 JSON 파일 경로")
    parser.add_argument("--model-path", type=str, default=None)

    # 카메라 / FPS 관련
    parser.add_argument("--cam-index",   type=int,   default=1,    help="웹캠 인덱스 (기본 1)")
    parser.add_argument("--grab-fps",    type=float, default=None, help="카메라 프레임 가져오기 FPS (기본 30)")
    parser.add_argument("--process-fps", type=float, default=None, help="YOLO 추론 FPS (기본 10)")

    # 자세 판정 임계값들 - 모두 위 전역값을 default 로 사용
    parser.add_argument("--kpt-conf-th",                      type=float, default=KPT_CONF_TH)
    parser.add_argument("--standing-knee-angle-th",           type=float, default=STANDING_KNEE_ANGLE_TH)
    parser.add_argument("--sitting-knee-angle-th",            type=float, default=SITTING_KNEE_ANGLE_TH)
    parser.add_argument("--lying-box-aspect-th",              type=float, default=LYING_BOX_ASPECT_TH)
    parser.add_argument("--torso-horizontal-th",              type=float, default=TORSO_HORIZONTAL_TH)
    parser.add_argument("--torso-vertical-th",                type=float, default=TORSO_VERTICAL_TH)
    parser.add_argument("--lying-score-th",                   type=float, default=LYING_SCORE_TH)
    parser.add_argument("--vertical-score-th",                type=float, default=VERTICAL_SCORE_TH)
    parser.add_argument("--head-hip-y-ratio-th",              type=float, default=HEAD_HIP_Y_RATIO_TH)
    parser.add_argument("--linearity-ratio-th",               type=float, default=LINEARITY_RATIO_TH)
    parser.add_argument("--standing-hip-above-knee-ratio-th", type=float, default=STANDING_HIP_ABOVE_KNEE_RATIO_TH)

    # 저장 간격
    parser.add_argument("--frame-save-interval-sec", type=float, default=FRAME_SAVE_INTERVAL_SEC)

    args = parser.parse_args()

    # 1) JSON 설정 파일 먼저 읽기
    cfg = load_config_json(args.config)

    # 2) 우선순위에 따라 최종 임계값 결정 (args > config json > 기본값)
    KPT_CONF_TH                      = pick(cfg, "kpt_conf_th",                       args.kpt_conf_th,                       KPT_CONF_TH)
    STANDING_KNEE_ANGLE_TH           = pick(cfg, "standing_knee_angle_th",            args.standing_knee_angle_th,            STANDING_KNEE_ANGLE_TH)
    SITTING_KNEE_ANGLE_TH            = pick(cfg, "sitting_knee_angle_th",             args.sitting_knee_angle_th,             SITTING_KNEE_ANGLE_TH)
    LYING_BOX_ASPECT_TH              = pick(cfg, "lying_box_aspect_th",               args.lying_box_aspect_th,               LYING_BOX_ASPECT_TH)
    TORSO_HORIZONTAL_TH              = pick(cfg, "torso_horizontal_th",               args.torso_horizontal_th,               TORSO_HORIZONTAL_TH)
    TORSO_VERTICAL_TH                = pick(cfg, "torso_vertical_th",                 args.torso_vertical_th,                 TORSO_VERTICAL_TH)
    LYING_SCORE_TH                   = pick(cfg, "lying_score_th",                    args.lying_score_th,                    LYING_SCORE_TH)
    VERTICAL_SCORE_TH                = pick(cfg, "vertical_score_th",                 args.vertical_score_th,                 VERTICAL_SCORE_TH)
    HEAD_HIP_Y_RATIO_TH              = pick(cfg, "head_hip_y_ratio_th",               args.head_hip_y_ratio_th,               HEAD_HIP_Y_RATIO_TH)
    LINEARITY_RATIO_TH               = pick(cfg, "linearity_ratio_th",                args.linearity_ratio_th,                LINEARITY_RATIO_TH)
    STANDING_HIP_ABOVE_KNEE_RATIO_TH = pick(cfg, "standing_hip_above_knee_ratio_th",  args.standing_hip_above_knee_ratio_th,  STANDING_HIP_ABOVE_KNEE_RATIO_TH)
    FRAME_SAVE_INTERVAL_SEC          = pick(cfg, "frame_save_interval_sec",           args.frame_save_interval_sec,           FRAME_SAVE_INTERVAL_SEC)

    # FPS도 동일한 우선순위 규칙으로
    grab_fps    = pick(cfg, "grab_fps",    args.grab_fps,    30.0)
    process_fps = pick(cfg, "process_fps", args.process_fps, 10.0)

    # 모델 경로 결정
    model_path = pick(cfg, "model_path", args.model_path, "yolov8n-pose.pt")

    # 모델 파일이 없으면 기본 모델 이름으로 폴백
    # (ultralytics는 가중치 이름만 주면 자동 다운로드도 시도함)
    if not os.path.exists(model_path):
        print(f"파일을 찾을 수 없습니다: {model_path}")
        print("기본 모델인 yolov8n-pose.pt를 사용합니다.")
        model_path = "yolov8n-pose.pt"

    try:
        model = YOLO(model_path, task="pose")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit(1)

    # 출력 폴더 초기화 - 매 실행마다 깨끗하게 시작 (이전 결과 삭제)
    output_dir = "./pose_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 프로세서 생성 → 실행 → 저장
    processor = YOLOPoseProcessor(
        model,
        output_dir,
        cam_index=args.cam_index,
        grab_fps=grab_fps,
        process_fps=process_fps,
    )

    processor.run()
    processor.save_output()

    print("프로그램이 정상적으로 종료되었습니다.")
    sys.exit(0)


if __name__ == "__main__":
    main()


# =============================================================================
# 다른 모듈(특히 amr_detect.py)에서 import 해 사용하는 어댑터 클래스
#
# 주의: 원본 코드는 이 클래스가 if __name__ 블록 *아래* 위치하지만,
#       파이썬은 import 시 모듈 전체를 실행하므로 이 위치에 있어도
#       PoseClassifier 자체는 정상적으로 import 가능합니다.
#       (단독 실행 시에는 main() 안에서 sys.exit() 가 호출되어
#        이 클래스 정의가 실행되기는 하지만, 어차피 그 시점엔 의미 없음)
# =============================================================================
class PoseClassifier:
    def classify(self, kpt, box):
        # 함수형 분류기 classify_pose 를 객체 메서드로 감싼 얇은 래퍼
        return classify_pose(kpt, box)
