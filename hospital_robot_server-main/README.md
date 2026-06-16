# 🏥 병원용 로봇 원격 제어 웹 서버

Flask + ROS2 + MySQL + YOLOv8 기반 병원 로봇 원격 제어 시스템

> ✅ ROS2 · MySQL · YOLO가 없어도 **Mock 모드**로 자동 실행됩니다.

---

## 📋 목차

1. [프로젝트 구조](#-프로젝트-구조)
2. [주요 기능](#-주요-기능)
3. [실행 환경](#-실행-환경)
4. [설치 및 실행](#-설치-및-실행)
5. [MySQL 설정](#-mysql-설정)
6. [LOCATION_DB 설정](#-location_db-설정)
7. [웹 페이지 및 API](#-웹-페이지-및-api)
8. [데이터베이스 스키마](#-데이터베이스-스키마)
9. [ROS2 토픽](#-ros2-토픽)
10. [실행 모드](#-실행-모드)
11. [문제 해결](#-문제-해결)

---

## 📁 프로젝트 구조

```
hospital_robot_server/
├── app.py                  # 메인 서버 (Flask + ROS2 통합)
├── mysql_crud.py           # MySQL CRUD 클래스
├── mysql_database.py       # MySQL 연결 기반 클래스
├── yolov8s.pt              # YOLOv8s 사전 학습 모델 (Person Counter)
├── virtual_line.json       # 가상 라인 좌표 (입출입 감지용)
├── requirements.txt        # Python 패키지 목록
├── ecosystem.config.cjs    # PM2 실행 설정
├── static/
│   └── emergency_popup.js  # 전체 페이지 공통 응급 알림 팝업
└── templates/
    ├── r_login_center.html     # 로그인 페이지
    ├── r_index.html            # 대시보드 (병동·환자 DB 연동)
    ├── main_dashboard.html     # 대시보드 (r_index.html과 동일)
    ├── patient.html            # 환자 및 병실 관리
    ├── robot_plan.html         # 순찰 계획 관리 (병동 상태 연동)
    ├── patrol_control.html     # 순찰 실행 (DB 계획 로드)
    ├── robot_status.html       # 로봇 상태 모니터링
    ├── camera.html             # 카메라 스트리밍
    ├── camera_error.html       # 카메라 오류 페이지
    ├── emergency_patient.html  # 응급 환자 현황
    └── msg_test.html           # ROS2 메시지 테스트
```

---

## ✨ 주요 기능

### 🗺️ 대시보드 (`/main`)
- **병동 상태 카드** — `rooms` 테이블 실시간 연동
  - 운영 중 / 전체 병동 수, safe · warning · danger 배지
- **현재 환자 카드** — `patients` 테이블 인원 수 실시간 표시
- **환자 상태 분포** — 정상 / 주의 / 응급 막대 바차트
- **병동 현황 패널** — 병동별 🟢 운영 중 / ⚫ 중단 배지 목록
- **로봇 배터리** — `/api/robot/status` 3초 폴링
- **상태 변화 자동 알림** — 병동 상태 변화, 환자 증감 시 알림 피드

### 📋 순찰 계획 (`/robot_plan`) — 병동 상태 동적 연동
- **병동 운영 배너** — 페이지 상단에 현재 병동 운영/중단 상태 표시
- **경로 선택 드롭다운** — 중단 병동 위치 `disabled` + 🚫 표시
- **비활성 위치 선택 시** — ⚠️ 경고 박스 표시, 추가 버튼 자동 비활성화
- **계획 목록 테이블** — 병동 상태 열 추가 (🟢 운영 중 / 🚫 중단 배지)
- **중단 병동 행** — 회색 흐림 처리, ▶ 패트롤 버튼 disabled
- **패트롤 미리보기** — 중단 위치 ~~취소선~~ 배지로 시각화
- **패트롤 실행** — 중단 병동 위치 자동 제외, 건너뛴 수 토스트 표시

### 🤖 순찰 실행 (`/patrol_control`)
- **현재 순찰 리스트** — 1초 폴링, 이름·좌표·감지 배지 표시
- **위치 추가 카드** — LOCATION_DB 기반 버튼 (🟢 감지 / ⚪ 이동)
- **DB 계획 카드** — `plan_table`에서 로봇별 탭으로 그룹화 표시
  - 전체 / 로봇 1 / 로봇 2 탭 전환
  - 항목별 `+ 추가` 버튼, 현재 탭 전체 실행 버튼
- **전체 비우기** — 순찰 리스트 초기화

### 🚨 코드 블루 (`/api/robot/code_blue`)
- robot1 현재 위치(AMCL 또는 Odometry)를 robot6 출동 좌표로 전송
- Nav2 `NavigateToPose` 액션으로 robot6 자율 주행
- Mock 모드에서도 좌표 반환 (테스트 가능)

### 🛑 응급 알림 팝업 (전체 페이지 공통)
- 모든 페이지에서 `/api/msg_test` 2초 폴링
- 위급상황 감지 시 팝업 표시 + 🚑 코드 블루 출동 버튼
- 확인(/api/emergency/confirm) 전까지 위급 상태 유지
- Mock 테스트: `/api/emergency/trigger`로 강제 트리거

---

## ⚙️ 실행 환경

| 항목 | 버전 | 필수 여부 |
|------|------|-----------|
| Python | 3.10 이상 | ✅ 필수 |
| ROS2 | Humble / Iron | 🔶 권장 (없으면 Mock) |
| MySQL | 8.0 이상 | 🔶 권장 (없으면 Mock) |
| YOLOv8 (ultralytics) | 8.0 이상 | 🔶 권장 (없으면 비활성화) |
| OpenCV (cv2) | 4.8 이상 | 🔶 권장 (없으면 비활성화) |
| Node.js + PM2 | 18 이상 | 🔶 권장 (없으면 직접 실행) |

---

## 🛠️ 설치 및 실행

### 1. Python 패키지 설치

```bash
cd hospital_robot_server
pip install -r requirements.txt
```

`requirements.txt` 주요 항목:
```
Flask>=2.3.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
PyMySQL>=1.1.0
```

### 2. ROS2 환경 소싱 (ROS2 설치된 경우)

```bash
source /opt/ros/humble/setup.bash
# 워크스페이스가 있다면 추가 소싱
source ~/your_ws/install/setup.bash
```

### 3. 서버 실행

#### 방법 A — PM2 백그라운드 실행 (권장)

```bash
npm install -g pm2   # PM2 최초 설치

cd hospital_robot_server
pm2 start ecosystem.config.cjs

pm2 list                              # 상태 확인
pm2 logs hospital-robot --nostream   # 로그 확인
pm2 restart hospital-robot           # 재시작
pm2 stop hospital-robot              # 중지
```

`ecosystem.config.cjs` 기본 환경변수:

```javascript
env: {
  SECRET_KEY: 'hospital-robot-secret-2024',  // Flask 세션 키 (고정값 필수)
  APP_USER:   'admin',                        // 폴백 로그인 ID
  APP_PASS:   'admin1234'                     // 폴백 로그인 PW
}
```

#### 방법 B — 직접 실행 (개발용)

```bash
cd hospital_robot_server
export SECRET_KEY="hospital-robot-secret-2024"
export APP_USER="admin"
export APP_PASS="admin1234"
python3 app.py
```

접속: `http://localhost:5002`  
로그인: `admin` / `admin1234`

---

## 🗄️ MySQL 설정

### 데이터베이스 및 테이블 생성

```bash
sudo mysql -u root
```

```sql
CREATE DATABASE hospital_robot_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '1234';
FLUSH PRIVILEGES;
USE hospital_robot_db;
```

```sql
-- 사용자
CREATE TABLE users (
    user_id  INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(50)
);

-- 환자 (patient_state: 0=정상, 1=주의, 2=응급)
CREATE TABLE patients (
    patient_id       INT PRIMARY KEY AUTO_INCREMENT,
    room_name        VARCHAR(50),
    patient_position VARCHAR(255),
    patient_state    INT
);

-- 병동 (existence: TRUE=운영 중, FALSE=중단)
CREATE TABLE rooms (
    room_id   INT PRIMARY KEY AUTO_INCREMENT,
    room_name VARCHAR(50),
    existence BOOLEAN
);

-- 응급
CREATE TABLE emergency (
    emergency_id     INT PRIMARY KEY AUTO_INCREMENT,
    patient_position VARCHAR(255),
    patient_state    INT
);

-- 순찰 계획 (fulfill: FALSE=이동만, TRUE=주변 감지)
CREATE TABLE plan_table (
    plan_ID   INT PRIMARY KEY AUTO_INCREMENT,
    robot_num INT NOT NULL,
    room_name VARCHAR(50),
    x_pose    FLOAT,
    y_pose    FLOAT,
    yaw       FLOAT,
    fulfill   BOOLEAN DEFAULT 0
);
```

### 초기 데이터 삽입

```sql
INSERT INTO users (username, password) VALUES ('admin', 'admin1234');

INSERT INTO rooms (room_name, existence) VALUES
    ('병동1', TRUE),
    ('병동2', TRUE);

INSERT INTO patients (room_name, patient_position, patient_state) VALUES
    ('병동1', 'A-01', 0),
    ('병동2', 'B-01', 1);
```

### app.py 연결 정보 수정 (약 183번째 줄)

```python
con = MySQL_Execute('localhost', 'root', '1234', 'hospital_robot_db')
```

---

## 📍 LOCATION_DB 설정

`app.py`의 `LOCATION_DB`에서 로봇 순찰 위치를 정의합니다.

### 형식: `"위치명": (x, y, yaw, detect)`

| 항목 | 타입 | 설명 |
|------|------|------|
| `x` | float | 맵 기준 X 좌표 (미터) |
| `y` | float | 맵 기준 Y 좌표 (미터) |
| `yaw` | float | 도착 방향각 (라디안) |
| `detect` | bool | `True`=도착 후 YOLO 주변 감지 / `False`=이동만 |

```python
LOCATION_DB = {
    # ── 안내 구역 ────────────────────────────────────────
    "안내데스크":        (-1.843,  0.098,  3.059, False),
    "안내복도_정면":     (-2.708,  0.149,  3.060, False),
    "안내복도_좌측":     (-2.785,  0.128, -1.624, False),
    "안내복도_우측":     (-2.814,  0.081,  1.900, False),
    # ── 병동1 ────────────────────────────────────────────
    "병동1_복도입구":    (-2.924, -1.475, -1.644, False),
    "병동1_내부":        (-2.093, -1.596, -0.058, True),   # ← YOLO 감지
    "병동1_침대앞":      (-0.348, -1.733, -3.076, True),   # ← YOLO 감지
    # ── 병동2 ────────────────────────────────────────────
    "병동2_복도입구_우": (-2.769,  1.400,  1.421, False),
    "병동2_복도_후면":   (-2.767,  1.448, -0.121, False),
    "병동2_내부":        (-1.774,  1.585, -0.116, True),   # ← YOLO 감지
    "병동2_침대앞_후면": (-0.181,  1.357,  3.113, True),   # ← YOLO 감지
    "병동2_침대앞_좌측": (-1.085,  3.497, -1.763, True),   # ← YOLO 감지
    # ── 복도 끝 ──────────────────────────────────────────
    "복도끝_좌측":       (-2.354,  3.890, -1.781, True),   # ← YOLO 감지
}
```

### ⚠️ LOCATION_DB 하드코딩 제거 (v2.0 이후)

> **v2.0부터 `LOCATION_DB` 딕셔너리가 `app.py`에서 완전히 제거되었습니다.**  
> 모든 위치 데이터는 `plan_table` DB에서 동적으로 로드됩니다.

#### 위치 추가 방법 (기존 LOCATION_DB → plan_table 직접 등록)
1. **순찰 계획 페이지(`/robot_plan`)** → 새 순찰 계획 추가 폼 사용
2. **직접 API 호출**: `POST /robot_plan` (form data: robot_num, room_name, pose_x, pose_y, yaw, fulfill)

#### 병동 상태 자동 연동 규칙

위치명이 `병동1_*` 형태이면 → `rooms` 테이블의 `병동1` `existence` 값으로 활성 여부 자동 판단

| 조건 | 결과 |
|------|------|
| `병동2` existence = FALSE | `병동2_*` 위치 전체 비활성화 |
| 병동 소속 없는 위치 (안내데스크, 복도끝 등) | 항상 활성 |

> RViz2 `2D Pose Estimate`로 확인한 맵 좌표계 값을 사용하세요.

---

## 🌐 웹 페이지 및 API

### 웹 페이지

| URL | 페이지명 | 주요 기능 |
|-----|---------|-----------|
| `/login` | 로그인 | 세션 인증 |
| `/main` | 대시보드 | 병동·환자 현황, 배터리, 알림 |
| `/patient` | 환자 관리 | 환자·병실 등록·삭제 |
| `/robot_plan` | 순찰 계획 | plan_table CRUD, 병동 상태 연동 |
| `/patrol_control` | 순찰 실행 | 실시간 패트롤 리스트, DB 계획 로드 |
| `/robot_status` | 로봇 상태 | robot1 / robot6 모니터링 |
| `/camera` | 카메라 | OAK-D 스트리밍 |
| `/emergency_patient` | 응급 환자 | 응급 현황 |
| `/msg_test_page` | Topic 테스트 | ROS2 메시지 발행 테스트 |

### 대시보드 API

| 메서드 | URL | 설명 |
|--------|-----|------|
| GET | `/api/dashboard/summary` | 병동(rooms) + 환자(patients) 집계 |
| GET | `/api/locations` | plan_table 기반 위치 목록 (ward_active 포함) |

응답 예시:
```json
{
  "success": true,
  "rooms": {
    "total": 2, "active": 1,
    "status": "warning", "label": "1/2 운영 중",
    "list": [
      {"room_id": 1, "room_name": "병동1", "active": true},
      {"room_id": 2, "room_name": "병동2", "active": false}
    ]
  },
  "patients": {
    "total": 3, "normal": 2, "caution": 1, "emergency": 0
  }
}
```

### 순찰 계획 API

| 메서드 | URL | 설명 |
|--------|-----|------|
| GET/POST/DELETE | `/robot_plan` | plan_table CRUD |
| GET | `/api/plan/list` | 전체 계획 조회 |
| POST | `/api/plan/run` | 계획 → patrol_list 적재 (비활성 병동 자동 제외) |
| POST | `/api/plan/delete` | 계획 삭제 + patrol_list 동기화 |

`/api/plan/run` 요청 / 응답:
```json
// 요청
{ "robot_num": 1 }   // 0 = 전체

// 응답
{
  "success": true,
  "count": 2,
  "list": [{"name": "병동1_내부", "x": -2.093, "y": -1.596, "detect": true}, ...],
  "skipped": ["병동2_내부"]   // 중단 병동으로 제외된 항목
}
```

### 패트롤 API

| 메서드 | URL | 설명 |
|--------|-----|------|
| GET | `/api/patrol/status` | 현재 patrol_list 조회 |
| POST | `/api/patrol/add` | 위치 추가 `{"name": "병동1_내부"}` |
| POST | `/api/patrol/remove` | 인덱스 삭제 `{"index": 0}` |
| POST | `/api/patrol/set` | 리스트 전체 교체 `{"list": [...]}` |
| POST | `/api/patrol/clear` | 전체 초기화 |

### 로봇 제어 API

| 메서드 | URL | 설명 |
|--------|-----|------|
| GET | `/api/robot/status` | robot1 배터리·위치·긴급 상태 |
| GET | `/api/robot6/status` | robot6 배터리·위치 |
| POST | `/api/robot/cmd` | robot1 속도 명령 `{linear, angular}` |
| POST | `/api/robot/code_blue` | robot6 코드 블루 출동 (robot1 위치 전달) |
| POST | `/api/server/patrol_cmd` | 순찰 명령 ROS2 전송 |

### 응급 API

| 메서드 | URL | 설명 |
|--------|-----|------|
| GET | `/api/msg_test` | 응급 상태 폴링 (2초마다) |
| POST | `/api/emergency/confirm` | 응급 팝업 확인 처리 (상태 리셋) |
| POST | `/api/emergency/trigger` | 응급 상황 강제 트리거 (Mock 테스트용) |
| GET | `/api/emergency` | 응급 목록 조회 |
| POST | `/api/emergency` | 응급 등록 |
| DELETE | `/api/emergency/<id>` | 응급 삭제 |

### 환자 / 병실 API

| 메서드 | URL | 설명 |
|--------|-----|------|
| GET | `/api/patient/status` | 환자 ROS2 상태·위치 |
| POST | `/api/patient` | 환자 등록 |
| DELETE | `/api/patient/<id>` | 환자 삭제 |
| POST | `/api/room` | 병실 등록 |
| DELETE | `/api/room/<id>` | 병실 삭제 |

### 카메라 / 스트리밍

| URL | 설명 |
|-----|------|
| `/stream/robot1/camera` | OAK-D 카메라 MJPEG 스트림 |
| `/stream/usb_camera` | USB 웹캠 MJPEG 스트림 (YOLO 오버레이) |
| `/api/camera/status` | 카메라 가용 여부 |
| `/api/person_count` | 입출입 인원 카운트 |

---

## 🗄️ 데이터베이스 스키마

```
hospital_robot_db
├── users      (user_id, username, password)
├── patients   (patient_id, room_name, patient_position, patient_state)
├── rooms      (room_id, room_name, existence)
├── emergency  (emergency_id, patient_position, patient_state)
└── plan_table (plan_ID, robot_num, room_name, x_pose, y_pose, yaw, fulfill)
```

**patient_state 값:**

| 값 | 의미 |
|----|------|
| `0` | 정상 |
| `1` | 주의 |
| `2` | 응급 |

**rooms.existence 값:**

| 값 | 의미 | 순찰 계획 영향 |
|----|------|----------------|
| `TRUE` | 운영 중 🟢 | 해당 병동 위치 활성화 |
| `FALSE` | 중단 🚫 | 해당 병동 위치 비활성화, 패트롤 실행 시 자동 제외 |

**plan_table.fulfill 값:**

| 값 | 의미 |
|----|------|
| `FALSE` (0) | 이동만 — NavigateToPose 도착 후 별도 동작 없음 |
| `TRUE` (1) | 주변 감지 — 도착 후 YOLOv8 감지 실행 |

---

## 🤖 ROS2 토픽

### 구독 (Subscribe)

| 토픽 | 메시지 타입 | 용도 |
|------|------------|------|
| `/robot1/battery_state` | `sensor_msgs/BatteryState` | robot1 배터리 |
| `/robot1/odom` | `nav_msgs/Odometry` | robot1 오도메트리 |
| `/robot1/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | robot1 AMCL 위치 |
| `/robot1/oakd/rgb/image_raw/compressed` | `sensor_msgs/CompressedImage` | OAK-D 카메라 |
| `/robot1/emergency_alert` | `std_msgs/String` | 긴급 알림 수신 |
| `/robot1/search_done` | `std_msgs/String` | 탐색 완료 수신 |
| `/robot6/battery_state` | `sensor_msgs/BatteryState` | robot6 배터리 |
| `/robot6/odom` | `nav_msgs/Odometry` | robot6 오도메트리 |
| `/robot6/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | robot6 AMCL 위치 |
| `/patient/status` | `std_msgs/String` | 환자 상태 수신 |
| `/patient/location` | `std_msgs/Float32MultiArray` | 환자 위치 수신 |
| `/msg_test` | `std_msgs/String` | 메시지 테스트 수신 |

### 퍼블리시 (Publish)

| 토픽 | 메시지 타입 | 용도 |
|------|------------|------|
| `/robot1/cmd_vel` | `geometry_msgs/Twist` | robot1 속도 명령 |
| `/patrol_cmd` | `std_msgs/String` | 패트롤 명령 (JSON) |
| `/send_msg` | `std_msgs/String` | 메시지 전송 |

### Nav2 Action

| 액션 서버 | 타입 | 용도 |
|-----------|------|------|
| `/robot6/navigate_to_pose` | `nav2_msgs/NavigateToPose` | robot6 자율 주행 (코드 블루) |

---

## 🔄 실행 모드

서버 시작 시 환경을 자동 감지하며, 없는 항목은 Mock 모드로 동작합니다.

| 기능 | 정상 모드 | Mock 모드 |
|------|-----------|-----------|
| 로그인 | MySQL `users` 테이블 | 인메모리 (`admin` / `admin1234`) |
| 환자·병실 CRUD | MySQL 실시간 반영 | 인메모리 임시 저장 |
| 로봇 상태 | ROS2 토픽 실시간 | 0.0 고정값 |
| 카메라 스트림 | OAK-D / USB 실제 영상 | 빈 프레임 |
| Person Counter | YOLOv8 실시간 감지 | 카운트 0 고정 |
| 패트롤 명령 | `/patrol_cmd` ROS2 전송 | 인메모리만 변경 |
| 코드 블루 | robot6 Nav2 실제 출동 | 좌표 반환 (이동 없음) |

**Mock 모드 인메모리 초기 데이터:**

```
users:     admin / admin1234
rooms:     병동1 (운영 중), 병동2 (중단)
patients:  101호 A-01 (정상), 102호 B-02 (주의)
emergency: 복도 3층 응급
```

---

## ❓ 문제 해결

| 증상 | 원인 | 해결 방법 |
|------|------|-----------|
| `[DB] MySQL 연결 실패` | MySQL 미실행 또는 연결 정보 오류 | `sudo systemctl start mysql` / app.py 연결 정보 확인 |
| `[ROS2] 미설치 → Mock 모드` | ROS2 미소싱 | `source /opt/ros/humble/setup.bash` |
| `카메라 index out of range` | USB 카메라 미연결 | 카메라 연결 또는 `PERSON_COUNTER_CAM` 변경 |
| `ultralytics 없음` | YOLO 미설치 | `pip install ultralytics` |
| 로그인 불가 (세션 만료) | `SECRET_KEY` 미고정 | `ecosystem.config.cjs`에 고정 `SECRET_KEY` 설정 확인 |
| 패트롤 실행 시 일부 제외 | 병동 `existence=FALSE` | 대시보드 또는 환자 페이지에서 병동 상태 확인 |
| 코드 블루 위치 0,0 반환 | AMCL/Odom 미수신 (Mock) | ROS2 연결 후 재시도 |
| 응급 팝업이 사라지지 않음 | 확인 버튼 미클릭 | 팝업의 `확인` 버튼 클릭 또는 `/api/emergency/confirm` POST |

---

## 📅 변경 이력

| 커밋 | 내용 |
|------|------|
| (최신) | LOCATION_DB 하드코딩 제거 → plan_table DB 완전 전환, /api/locations 신설 |
| `429dd12` | 병동 운영상태 → 순찰 계획 동적 연동 |
| `4b61f7e` | 대시보드 병동/환자 DB 연동 개편 |
| `5295d24` | plan/run 좌표 손실 버그 수정 |
| `9fe538a` | 탭명/LOCATION_DB 한글화 |
| `7333fd9` | 로봇 계획(plan_table) ↔ 패트롤 컨트롤 완전 연동 |
| `c6a99a5` | 코드 블루 기능 정상화 |
| `3eb6ff0` | 전체 페이지 공통 응급 알림 팝업 추가 |

---

*최종 업데이트: 2026-04-26 (LOCATION_DB → plan_table DB 완전 전환)*
