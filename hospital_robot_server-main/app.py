# =============================================================================
# 병원용 로봇 원격 제어 웹 서버 (Flask + ROS2 연동) — 통합본
# =============================================================================
# [통합 내용]
#   login.py   : robot6 제어, NavigateToPose(코드 블루), MultiThreadedExecutor,
#                msg_test/emergency/search 토픽, patrol_cmd, send_msg API
#   login(1).py: MySQL DB 로그인, 환자·병실 CRUD API, YOLO Person Counter,
#                plan_table DB 연동, 메인 페이지 DB 통계
#
# [실행 모드]
#   - ROS2 / YOLO / MySQL / cv2 미설치 환경에서도 자동으로 Mock 모드로 실행됩니다.
#   - 포트: 5002
# =============================================================================

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash,
    jsonify, Response
)
import threading
import time
import math
import os
import json

# ── OpenCV (없으면 Mock) ────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    print("[환경] cv2 / numpy 사용 가능")
except ImportError:
    CV2_AVAILABLE = False
    print("[환경] cv2 없음 → 카메라 스트리밍 비활성화")

# ── ultralytics YOLO (없으면 Mock) ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("[환경] ultralytics YOLO 사용 가능")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    print("[환경] ultralytics 없음 → Person Counter 비활성화")

# ── ROS2 (없으면 Mock 클래스로 대체) ────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from nav2_msgs.action import NavigateToPose
    from sensor_msgs.msg import BatteryState, CompressedImage
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
    from std_msgs.msg import String, Float32MultiArray
    ROS2_IMPORT_OK = True
    print("[환경] ROS2 패키지 사용 가능")
except ImportError:
    ROS2_IMPORT_OK = False
    print("[환경] ROS2 없음 → Mock 모드로 실행")

    # ── 최소 Mock 클래스 ──────────────────────────────────────────────────────
    class Node:
        def __init__(self, name): pass

    class ActionClient:
        def __init__(self, *a, **kw): pass
        def server_is_ready(self): return False
        def send_goal_async(self, g): pass

    class MultiThreadedExecutor:
        def add_node(self, n): pass
        def spin(self): pass

    class _MsgStub:
        data = None

    class Twist:
        class _L: x = 0.0
        class _A: z = 0.0
        linear  = _L()
        angular = _A()

    NavigateToPose  = None
    BatteryState    = None
    CompressedImage = None
    Odometry        = None
    PoseWithCovarianceStamped = None
    String          = _MsgStub
    Float32MultiArray = _MsgStub

# ── MySQL (없으면 Mock) ──────────────────────────────────────────────────────
try:
    from mysql_crud import MySQL_Execute
    MYSQL_AVAILABLE = True
    print("[환경] MySQL 모듈 사용 가능")
except Exception as e:
    MYSQL_AVAILABLE = False
    print(f"[환경] MySQL 없음 ({e}) → Mock DB 사용")

    class MySQL_Execute:
        """MySQL 없을 때 사용하는 최소 Mock 클래스"""
        def __init__(self, *a, **kw): pass
        def select_data(self, table_name='', columns='*', where=None, once=False):
            return None if once else []
        def insert_data(self, table_name, data): return None
        def update_data(self, table_name, data, where): pass
        def delete_data(self, table_name, where): pass
        def execute_query(self, query, params=None): return None
        def execute_querys(self, query, params=None): return []


# =============================================================================
# MySQL 데이터베이스 연결 (실패해도 서버 계속 실행)
# =============================================================================

class _MockDB:
    """MySQL 연결 실패 시 사용하는 완전 Mock DB"""
    # ── 인메모리 샘플 데이터 ──────────────────────────────────────────────────
    _data = {
        'users':     [{'user_id': 1, 'username': 'admin', 'password': 'admin1234'}],
        'patients':  [
            {'patient_id': 1, 'room_name': '101호', 'patient_position': 'A-01', 'patient_state': 0},
            {'patient_id': 2, 'room_name': '102호', 'patient_position': 'B-02', 'patient_state': 1},
        ],
        'rooms': [
            {'room_id': 1, 'room_name': '병동1', 'existence': True},
            {'room_id': 2, 'room_name': '병동2', 'existence': False},
        ],
        'emergency': [
            {'emergency_id': 1, 'patient_position': '복도 3층', 'patient_state': 2},
        ],
        'plan_table': [],
    }
    _id_seq = {'patients': 3, 'rooms': 3, 'emergency': 2, 'plan_table': 1}

    def _match(self, row, where):
        """단순 'col=val' 단일 조건 파싱"""
        if not where:
            return True
        try:
            col, val = [s.strip() for s in where.split('=', 1)]
            val = val.strip('"\'')
            cell = row.get(col)
            if isinstance(cell, bool):
                return cell == (val.lower() in ('1', 'true'))
            return str(cell) == str(val)
        except Exception:
            return True

    def select_data(self, table_name='', columns='*', where=None, once=False):
        rows = [r for r in self._data.get(table_name, []) if self._match(r, where)]
        return (rows[0] if rows else None) if once else rows

    def insert_data(self, table_name, data):
        tbl = self._data.setdefault(table_name, [])
        pk_map = {'patients': 'patient_id', 'rooms': 'room_id',
                  'emergency': 'emergency_id', 'plan_table': 'plan_ID', 'users': 'user_id'}
        pk = pk_map.get(table_name)
        if pk:
            new_id = self._id_seq.get(table_name, len(tbl) + 1)
            data = dict(data); data[pk] = new_id
            self._id_seq[table_name] = new_id + 1
        tbl.append(data)
        return data

    def update_data(self, table_name, data, where):
        for row in self._data.get(table_name, []):
            if self._match(row, where):
                row.update(data)

    def delete_data(self, table_name, where):
        self._data[table_name] = [
            r for r in self._data.get(table_name, [])
            if not self._match(r, where)
        ]

    def execute_query(self, query, params=None): return None
    def execute_querys(self, query, params=None): return []

try:
    con = MySQL_Execute('localhost', 'root', '1234', 'hospital_robot_db')
    print("[DB] MySQL 연결 성공")
except Exception as e:
    print(f"[DB] MySQL 연결 실패 ({e}) → Mock DB 사용")
    con = _MockDB()


# =============================================================================
# Person Counter 설정값
# =============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_COUNTER_MODEL = os.path.join(_BASE_DIR, "yolov8s.pt")
PERSON_COUNTER_CAM   = 0
PERSON_COUNTER_CONF  = 0.25
PERSON_COUNTER_LINE  = os.path.join(_BASE_DIR, "virtual_line.json")

person_counts      = {"enter": 0, "exit": 0}
person_counts_lock = threading.Lock()


class _LineTrack:
    """개인 트래커: 가상 선 교차 여부를 판단합니다."""

    def __init__(self, tid):
        self.tid       = tid
        self.prev_side = None

    def check_crossing(self, current_pt, line_pts):
        if not CV2_AVAILABLE:
            return None
        if line_pts is None or len(line_pts) < 2:
            return None
        import numpy as _np
        p1, p2 = _np.array(line_pts[0]), _np.array(line_pts[1])
        curr_p = _np.array(current_pt)
        side   = _np.sign(_np.cross(p2 - p1, curr_p - p1))
        if self.prev_side is None:
            self.prev_side = side
            return None
        res = None
        if self.prev_side != side and side != 0:
            if self.prev_side == -1 and side == 1:
                res = "EXIT"
                try:
                    con.update_data('rooms', {"existence": False}, 'room_name="병동2"')
                except Exception:
                    pass
            elif self.prev_side == 1 and side == -1:
                res = "ENTER"
                try:
                    con.update_data('rooms', {"existence": True}, 'room_name="병동2"')
                except Exception:
                    pass
            self.prev_side = side
        return res


# =============================================================================
# plan_table DB 헬퍼
# =============================================================================
def fetch_plan_list():
    try:
        return con.select_data(table_name='plan_table', columns='*') or []
    except Exception:
        return []

def create_plan_entry(form_data):
    """plan_table INSERT  ─  컬럼명을 실제 DDL과 일치 (plan_ID 제외, AUTO_INCREMENT)"""
    plan_data = {
        'robot_num': int(form_data.get('robot_num', 0)),
        'room_name': form_data.get('room_name', ''),
        'x_pose':    float(form_data.get('pose_x', 0.0)),
        'y_pose':    float(form_data.get('pose_y', 0.0)),
        'yaw':       float(form_data.get('yaw', 0.0)),
        'fulfill':   form_data.get('fulfill', 'false') in ('true', '1', 'True', '회전')
    }
    con.insert_data(table_name='plan_table', data=plan_data)

def delete_plan_entry(plan_id=None, room_name=None):
    """plan_ID(PK) 또는 room_name 으로 삭제"""
    if plan_id is not None:
        con.delete_data(table_name='plan_table', where=f'plan_ID={plan_id}')
    elif room_name:
        con.delete_data(table_name='plan_table', where=f'room_name="{room_name}"')


# =============================================================================
# Flask 앱 초기화
# =============================================================================
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

USERNAME = os.getenv('APP_USER', 'user')
PASSWORD = os.getenv('APP_PASS', 'password')

_lock = threading.Lock()

# =============================================================================
# 공유 상태 딕셔너리
# =============================================================================
robot_data = {
    'battery_pct': 0.0, 'battery_voltage': 0.0,
    'odom_x': 0.0, 'odom_y': 0.0, 'odom_yaw': 0.0,
    'amcl_x': 0.0, 'amcl_y': 0.0, 'amcl_yaw': 0.0,
    'amcl_active': False
}
robot6_data = {
    'battery_pct': 0.0, 'battery_voltage': 0.0,
    'odom_x': 0.0, 'odom_y': 0.0, 'odom_yaw': 0.0,
    'amcl_x': 0.0, 'amcl_y': 0.0, 'amcl_yaw': 0.0,
    'amcl_active': False
}
patient_data = {'status': 'safe', 'location_x': 0.0, 'location_y': 0.0}
camera_data  = {'available': False}
latest_frame     = None
latest_usb_frame = None
test_message     = {"emergency": "대기 중", "search": "대기 중"}

_STATE_INT_TO_STR = {0: '일반', 1: '위험군', 2: '고위험군'}
_STATE_STR_TO_INT = {'일반': 0, '위험군': 1, '고위험군': 2}


# =============================================================================
# 유틸리티
# =============================================================================
def quat_to_yaw(q):
    yaw = math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z)
    )
    return round(math.degrees(yaw), 1)


# =============================================================================
# ROS2 노드 (ROS2 설치된 환경에서만 동작)
# =============================================================================
class RobotNode(Node):
    def __init__(self):
        super().__init__('flask_robot_node')
        if not ROS2_IMPORT_OK:
            return

        self.create_subscription(BatteryState, '/robot1/battery_state', self.battery_cb, 10)
        self.create_subscription(Odometry,     '/robot1/odom',          self.odom_cb,    10)
        self.create_subscription(PoseWithCovarianceStamped, '/robot1/amcl_pose', self.amcl_cb, 10)
        self.create_subscription(String,            '/patient/status',   self.status_cb,   10)
        self.create_subscription(Float32MultiArray, '/patient/location', self.location_cb, 10)
        self.create_subscription(CompressedImage,
            '/robot1/oakd/rgb/image_raw/compressed', self.camera_cb, 10)
        self.create_subscription(String, '/msg_test',               self.msg_test_cb,    10)
        self.create_subscription(String, '/robot1/emergency_alert', self.emergency_cb,   10)
        self.create_subscription(String, '/robot1/search_done',     self.search_done_cb, 10)
        self.create_subscription(BatteryState, '/robot6/battery_state', self.battery6_cb, 10)
        self.create_subscription(Odometry,     '/robot6/odom',          self.odom6_cb,    10)
        self.create_subscription(PoseWithCovarianceStamped,
            '/robot6/amcl_pose', self.amcl6_cb, 10)

        self.cmd_pub      = self.create_publisher(Twist,  '/robot1/cmd_vel', 10)
        self.plan_pub     = self.create_publisher(String, '/patrol_cmd',     10)
        self.msg_test_pub = self.create_publisher(String, '/send_msg',       10)
        self.robot6_nav_client = ActionClient(self, NavigateToPose, '/robot6/navigate_to_pose')

    def battery_cb(self, msg):
        with _lock:
            robot_data['battery_pct']     = round(msg.percentage * 100, 1)
            robot_data['battery_voltage'] = round(msg.voltage, 2)

    def odom_cb(self, msg):
        with _lock:
            robot_data['odom_x']   = round(msg.pose.pose.position.x, 3)
            robot_data['odom_y']   = round(msg.pose.pose.position.y, 3)
            robot_data['odom_yaw'] = quat_to_yaw(msg.pose.pose.orientation)

    def amcl_cb(self, msg):
        with _lock:
            robot_data['amcl_x']      = round(msg.pose.pose.position.x, 3)
            robot_data['amcl_y']      = round(msg.pose.pose.position.y, 3)
            robot_data['amcl_yaw']    = quat_to_yaw(msg.pose.pose.orientation)
            robot_data['amcl_active'] = True

    def status_cb(self, msg):
        with _lock: patient_data['status'] = msg.data

    def location_cb(self, msg):
        if len(msg.data) >= 2:
            with _lock:
                patient_data['location_x'] = round(msg.data[0], 3)
                patient_data['location_y'] = round(msg.data[1], 3)

    def camera_cb(self, msg):
        global latest_frame
        with _lock:
            latest_frame             = bytes(msg.data)
            camera_data['available'] = True

    def msg_test_cb(self, msg):
        with _lock: test_message['data'] = msg.data
        print(f"[msg_test] 수신: {msg.data}")

    def emergency_cb(self, msg):
        with _lock: test_message['emergency'] = msg.data
        print(f"[긴급] 수신: {msg.data}")

    def search_done_cb(self, msg):
        with _lock: test_message['search'] = msg.data
        print(f"[탐색] 수신: {msg.data}")

    def battery6_cb(self, msg):
        with _lock:
            robot6_data['battery_pct']     = round(msg.percentage * 100, 1)
            robot6_data['battery_voltage'] = round(msg.voltage, 2)

    def odom6_cb(self, msg):
        with _lock:
            robot6_data['odom_x']   = round(msg.pose.pose.position.x, 3)
            robot6_data['odom_y']   = round(msg.pose.pose.position.y, 3)
            robot6_data['odom_yaw'] = quat_to_yaw(msg.pose.pose.orientation)

    def amcl6_cb(self, msg):
        with _lock:
            robot6_data['amcl_x']      = round(msg.pose.pose.position.x, 3)
            robot6_data['amcl_y']      = round(msg.pose.pose.position.y, 3)
            robot6_data['amcl_yaw']    = quat_to_yaw(msg.pose.pose.orientation)
            robot6_data['amcl_active'] = True

    def pub_cmd(self, linear, angular):
        msg           = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def send_robot6_to(self, x, y, yaw_deg=0.0):
        if not self.robot6_nav_client.server_is_ready():
            print('[코드 블루] robot6 Nav2 action server 미준비')
            return False
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(x)
        goal.pose.pose.position.y    = float(y)
        goal.pose.pose.position.z    = 0.0
        yaw_rad = math.radians(float(yaw_deg))
        goal.pose.pose.orientation.z = math.sin(yaw_rad / 2)
        goal.pose.pose.orientation.w = math.cos(yaw_rad / 2)
        self.robot6_nav_client.send_goal_async(goal)
        return True


# =============================================================================
# ROS2 백그라운드 스레드
# =============================================================================
robot_node = None
ros2_ready = False

def ros2_thread():
    global robot_node, ros2_ready
    if not ROS2_IMPORT_OK:
        print("[ROS2] 미설치 → ROS2 스레드 비활성화")
        return
    try:
        import rclpy as _rclpy
        _rclpy.init()
        robot_node = RobotNode()
        ros2_ready = True
        executor   = MultiThreadedExecutor()
        executor.add_node(robot_node)
        executor.spin()
    except Exception as e:
        print(f'[ROS2] 초기화 실패: {e}')
        ros2_ready = False

threading.Thread(target=ros2_thread, daemon=True).start()


# =============================================================================
# USB 웹캠 / YOLO Person Counter 스레드
# =============================================================================
def usb_camera_thread():
    global latest_usb_frame
    if not CV2_AVAILABLE:
        print("[카메라] cv2 없음 → 카메라 스레드 비활성화")
        return

    if YOLO_AVAILABLE:
        model  = YOLO(PERSON_COUNTER_MODEL)
        tracks = {}
        v_line = None
        if os.path.exists(PERSON_COUNTER_LINE):
            with open(PERSON_COUNTER_LINE, "r") as f:
                v_line = json.load(f)

    cap = cv2.VideoCapture(PERSON_COUNTER_CAM)

    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(PERSON_COUNTER_CAM)
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.033)
            continue

        if YOLO_AVAILABLE:
            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                verbose=False, classes=[0],
                conf=PERSON_COUNTER_CONF, iou=0.5
            )
            vis = frame.copy()
            if v_line:
                cv2.line(vis, tuple(v_line[0]), tuple(v_line[1]), (0, 255, 0), 3)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.int().cpu().numpy()
                for box, tid in zip(boxes, ids):
                    feet_pt = (int((box[0] + box[2]) / 2), int(box[3]))
                    if tid not in tracks:
                        tracks[tid] = _LineTrack(tid)
                    if v_line:
                        crossed = tracks[tid].check_crossing(feet_pt, v_line)
                        if crossed == "ENTER":
                            with person_counts_lock: person_counts["enter"] += 1
                        elif crossed == "EXIT":
                            with person_counts_lock: person_counts["exit"] += 1
                    color = (255, 0, 255)
                    cv2.rectangle(vis, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), color, 2)
                    cv2.circle(vis, feet_pt, 5, (0, 0, 255), -1)
                    cv2.putText(vis, f"ID:{tid}",
                                (int(box[0]), int(box[1]) - 5), 1, 1, color, 2)

            with person_counts_lock:
                ent = person_counts["enter"]
                ext = person_counts["exit"]
            occ = max(0, ent - ext)
            cv2.putText(vis, f"Enter: {ent}  Exit: {ext}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"Occupancy: {occ}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            vis = frame

        _, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            latest_usb_frame = jpeg.tobytes()

threading.Thread(target=usb_camera_thread, daemon=True).start()


# =============================================================================
# JSON 헬퍼
# =============================================================================
def ok(data):
    return jsonify({'success': True, 'data': data})

def err(msg, code=400):
    return jsonify({'success': False, 'error': msg}), code


# =============================================================================
# 페이지 라우트
# =============================================================================
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('main'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        authenticated = False
        try:
            result = con.select_data(
                table_name='users', columns='username, password',
                where=f'username="{username}"', once=True
            )
            if result and username == result['username'] and password == result['password']:
                authenticated = True
        except Exception as e:
            print(f'[MySQL] 로그인 조회 실패, 폴백 사용: {e}')
            if username == USERNAME and password == PASSWORD:
                authenticated = True

        if not authenticated and username == USERNAME and password == PASSWORD:
            authenticated = True

        if authenticated:
            session['username'] = username
            return redirect(url_for('main'))
        flash('다시 입력하세요')
        return redirect(url_for('login'))
    return render_template('r_login_center.html')


@app.route('/main')
def main():
    if 'username' not in session:
        flash('Please log in first!', 'warning')
        return redirect(url_for('login'))
    room_count = room_total = patient_count = 0
    msgs = []
    try:
        room_count    = len(con.select_data(table_name='rooms',    columns='*', where='existence=true') or [])
        room_total    = len(con.select_data(table_name='rooms',    columns='*') or [])
        patient_count = len(con.select_data(table_name='patients', columns='*', where='patient_state > -1') or [])
        msgs          = con.select_data(table_name='emergency', columns='*') or []
    except Exception as e:
        print(f'[MySQL] 메인 조회 실패: {e}')
    return render_template('r_index.html', username=session['username'],
                           room_count=room_count, room_total=room_total,
                           patient_count=patient_count, msgs=msgs)


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/patient')
def patient():
    if 'username' not in session:
        return redirect(url_for('login'))
    patients = []
    rooms    = []
    try:
        raw_patients = con.select_data(table_name='patients', columns='*') or []
        raw_rooms    = con.select_data(table_name='rooms',    columns='*') or []
        patients = [{'id': p['patient_id'], 'room': p['room_name'],
                     'bed': p['patient_position'],
                     'status': _STATE_INT_TO_STR.get(p['patient_state'], '일반')}
                    for p in raw_patients]
        rooms    = [{'id': r['room_id'], 'pos': r['room_name'],
                     'inside': '있음' if r['existence'] else '없음'}
                    for r in raw_rooms]
    except Exception as e:
        print(f'[MySQL] 환자 조회 실패: {e}')
    return render_template('patient.html', username=session['username'],
                           patients=patients, rooms=rooms)


@app.route('/emergency_patient')
def emergency_patient():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('emergency_patient.html', username=session['username'])


@app.route('/camera')
def camera():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('camera.html', username=session['username'])


@app.route('/camera_error')
def camera_error():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('camera_error.html', username=session['username'])


@app.route('/robot_plan', methods=['GET', 'POST', 'DELETE'])
def robot_plan():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'GET':
        # rooms DB에서 운영 중인 병동 이름 목록 수집
        try:
            all_rooms = con.select_data(table_name='rooms', columns='*') or []
        except Exception:
            all_rooms = []
        # 운영 중인 병동 이름 집합 (existence=True인 것)
        active_wards = {
            r['room_name']
            for r in all_rooms
            if r.get('existence') in (True, 1, '1', 'true')
        }
        # 병동 전체 이름 집합 (운영여부 무관)
        all_ward_names = {r['room_name'] for r in all_rooms}

        def _ward_active(loc_name: str) -> bool:
            """위치명에 포함된 병동이 운영 중인지 확인.
            rooms DB에 등록된 병동이 없는 위치(복도끝 등)는 항상 활성."""
            for ward in all_ward_names:
                if loc_name.startswith(ward):
                    return ward in active_wards
            return True  # 소속 병동 없음 → 항상 활성

        # plan_table에서 위치 마스터 데이터 동적 로드 (LOCATION_DB 하드코딩 제거)
        raw_plans = fetch_plan_list()
        # 위치 드롭다운 목록: plan_table의 unique room_name 기준
        seen_names = set()
        locations = []
        for p in raw_plans:
            name = p.get('room_name', '')
            if name and name not in seen_names:
                seen_names.add(name)
                locations.append({
                    'name':        name,
                    'x':           float(p.get('x_pose', 0.0)),
                    'y':           float(p.get('y_pose', 0.0)),
                    'yaw':         float(p.get('yaw', 0.0)),
                    'detect':      bool(p.get('fulfill', False)),
                    'ward_active': _ward_active(name),
                })

        # plan_list에도 ward_active 플래그 추가
        plan_list_ext = []
        for p in raw_plans:
            p2 = dict(p)
            p2['ward_active'] = _ward_active(p.get('room_name', ''))
            plan_list_ext.append(p2)

        return render_template('robot_plan.html', username=session['username'],
                               plan_list=plan_list_ext,
                               locations=locations,
                               active_wards=sorted(active_wards),
                               all_ward_names=sorted(all_ward_names))
    if request.method == 'POST':
        try:
            create_plan_entry(request.form)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
        return jsonify({'success': True})
    if request.method == 'DELETE':
        # plan_ID 우선, 없으면 room_name 으로 삭제
        data = request.get_json(silent=True) or {}
        plan_id   = data.get('plan_id') or request.form.get('plan_id')
        room_name = data.get('room_name') or request.form.get('room_name', '').strip()
        if not plan_id and not room_name:
            return jsonify({'success': False, 'message': 'plan_id 또는 room_name 필요'}), 400
        try:
            delete_plan_entry(plan_id=plan_id, room_name=room_name)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
        return jsonify({'success': True})


@app.route('/robot_status')
def robot_status():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('robot_status.html', username=session['username'])


@app.route('/msg_test_page')
def msg_test_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('msg_test.html', username=session['username'])


# =============================================================================
# 영상 스트리밍
# =============================================================================
def _frame_generator():
    while True:
        with _lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/stream/robot1/camera')
def stream_camera():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def _usb_frame_generator():
    while True:
        with _lock:
            frame = latest_usb_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/stream/usb_camera')
def stream_usb_camera():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(_usb_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================================================
# REST API
# =============================================================================
@app.route('/api/robot/status')
def api_robot_status():
    with _lock:
        snapshot              = dict(robot_data)
        snapshot['emergency'] = test_message.get('emergency', '대기 중')
        snapshot['search']    = test_message.get('search',    '대기 중')
    return ok(snapshot)


@app.route('/api/robot6/status')
def api_robot6_status():
    with _lock:
        snapshot = dict(robot6_data)
    return ok(snapshot)


@app.route('/api/robot/cmd', methods=['POST'])
def api_robot_cmd():
    if not ros2_ready or robot_node is None:
        return err('ROS2 노드가 준비되지 않았습니다.', 503)
    data = request.get_json(silent=True)
    if not data:
        return err('요청 본문이 올바르지 않습니다.', 400)
    try:
        linear  = float(data.get('linear',  0.0))
        angular = float(data.get('angular', 0.0))
    except (TypeError, ValueError):
        return err('linear / angular 값이 올바르지 않습니다.', 400)
    robot_node.pub_cmd(linear, angular)
    return ok({'linear': linear, 'angular': angular})


@app.route('/api/robot/code_blue', methods=['POST'])
def api_code_blue():
    """
    [POST] /api/robot/code_blue
    코드 블루 발생 시 robot6을 robot1의 현재 위치로 출동시킵니다.

    robot1의 AMCL 위치(amcl_active=True)를 우선 사용하고,
    AMCL이 비활성 상태면 오도메트리 위치로 대체합니다.

    ROS2 미연결(Mock 모드) 시에도 robot1 현재 좌표를 반환합니다.
    """
    if 'username' not in session:
        return err('로그인이 필요합니다.', 401)

    with _lock:
        if robot_data['amcl_active']:
            x, y, yaw = robot_data['amcl_x'], robot_data['amcl_y'], robot_data['amcl_yaw']
            source = 'amcl'
        else:
            x, y, yaw = robot_data['odom_x'], robot_data['odom_y'], robot_data['odom_yaw']
            source = 'odom'

    # ROS2 실제 연결된 경우: robot6 Nav2 action 발행
    if ros2_ready and robot_node is not None:
        if not robot_node.send_robot6_to(x, y, yaw):
            return err('robot6 Nav2 action server가 준비되지 않았습니다.', 503)
        print(f'[코드 블루] robot6 출동 명령 전송 → x={x}, y={y}, yaw={yaw} ({source})')
        return ok({'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 1),
                   'source': source, 'mode': 'ros2'})

    # Mock 모드: 실제 Nav2 없이 좌표만 반환 (테스트용)
    print(f'[코드 블루 Mock] robot1 현재 위치 → x={x}, y={y}, yaw={yaw} ({source})')
    return ok({'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 1),
               'source': source, 'mode': 'mock'})


@app.route('/api/server/patrol_cmd', methods=['POST'])
def api_patrol_cmd():
    if not ros2_ready or robot_node is None:
        return err('ROS2 노드 연결 실패', 503)
    try:
        data    = request.get_json(silent=True) or {}
        payload = {
            "x": float(data.get('x', 0.0)), "y": float(data.get('y', 0.0)),
            "yaw": float(data.get('yaw', 0.0)), "detect": bool(data.get('detect', False)),
            "cmd": "START"
        }
        msg      = String()
        msg.data = json.dumps(payload)
        robot_node.plan_pub.publish(msg)
        return ok({'message': 'PC1으로 순찰 명령 전송 완료', 'payload': payload})
    except (TypeError, ValueError) as e:
        return err(f"데이터 형식 오류: {str(e)}", 400)
    except Exception as e:
        return err(f"서버 내부 오류: {str(e)}", 500)


@app.route('/api/robot/send_msg', methods=['POST'])
def api_send_test_msg():
    data       = request.get_json(silent=True) or {}
    target_msg = data.get('message', '')
    if robot_node and ros2_ready:
        msg      = String()
        msg.data = target_msg
        robot_node.msg_test_pub.publish(msg)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'ROS2 노드 없음'}), 503


@app.route('/api/robot/patrol', methods=['POST'])
def api_robot_patrol():
    data  = request.json or {}
    route = data.get('route', '')
    return jsonify({'status': f'{route} 순찰 시작'})


@app.route('/api/msg_test')
def api_msg_test():
    with _lock:
        emergency = test_message.get('emergency', '대기 중')
        search    = test_message.get('search',    '대기 중')
        # 위급상황 발생 시 환자 위치도 함께 전달 (리셋 없이 유지)
        location  = ''
        if emergency == '위급상황':
            loc_x = patient_data.get('location_x', 0.0)
            loc_y = patient_data.get('location_y', 0.0)
            if loc_x != 0.0 or loc_y != 0.0:
                location = f'환자 위치: ({loc_x:.2f}, {loc_y:.2f})'
    return jsonify({'success': True, 'data': {
        'emergency': emergency,
        'search':    search,
        'location':  location,
    }})


@app.route('/api/emergency/confirm', methods=['POST'])
def api_emergency_confirm():
    """[POST] /api/emergency/confirm — 클라이언트가 팝업 확인 후 호출, 위급상황 리셋"""
    if 'username' not in session:
        return err('로그인이 필요합니다.', 401)
    with _lock:
        test_message['emergency'] = '대기 중'
    return ok({'message': '위급상황 상태 초기화 완료'})


@app.route('/api/emergency/trigger', methods=['POST'])
def api_emergency_trigger():
    """[POST] /api/emergency/trigger — Mock 모드 테스트용: 위급상황 강제 트리거"""
    if 'username' not in session:
        return err('로그인이 필요합니다.', 401)
    data = request.get_json(silent=True) or {}
    loc_x = float(data.get('x', 0.0))
    loc_y = float(data.get('y', 0.0))
    with _lock:
        test_message['emergency'] = '위급상황'
        patient_data['location_x'] = loc_x
        patient_data['location_y'] = loc_y
    print(f'[위급상황 트리거] 강제 설정 → x={loc_x}, y={loc_y}')
    return ok({'message': '위급상황 트리거 완료', 'x': loc_x, 'y': loc_y})


@app.route('/api/patient/status')
def api_patient_status():
    with _lock:
        snapshot = dict(patient_data)
    return ok(snapshot)


@app.route('/api/patient', methods=['POST'])
def api_add_patient():
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'message': '요청 본문 오류'}), 400
    try:
        con.insert_data('patients', {
            'room_name': data['room'], 'patient_position': data['bed'],
            'patient_state': _STATE_STR_TO_INT.get(data['status'], 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True})


@app.route('/api/patient/<int:patient_id>', methods=['DELETE'])
def api_delete_patient(patient_id):
    if 'username' not in session:
        return jsonify({'success': False}), 401
    try:
        con.delete_data('patients', f'patient_id={patient_id}')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True})


@app.route('/api/room', methods=['POST'])
def api_add_room():
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'message': '요청 본문 오류'}), 400
    try:
        con.insert_data('rooms', {
            'room_name': data['pos'],
            'existence': 1 if data.get('inside') == '있음' else 0
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True})


@app.route('/api/room/<int:room_id>', methods=['DELETE'])
def api_delete_room(room_id):
    if 'username' not in session:
        return jsonify({'success': False}), 401
    try:
        con.delete_data('rooms', f'room_id={room_id}')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True})


# =============================================================================
# emergency 테이블 API
# =============================================================================
@app.route('/api/emergency', methods=['GET'])
def api_emergency_list():
    """응급 목록 조회"""
    try:
        rows = con.select_data(table_name='emergency', columns='*') or []
        result = [{
            'emergency_id':    r.get('emergency_id'),
            'patient_position': r.get('patient_position', ''),
            'patient_state':   r.get('patient_state', 0),
            'state_label':     _STATE_INT_TO_STR.get(r.get('patient_state', 0), '일반')
        } for r in rows]
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@app.route('/api/emergency', methods=['POST'])
def api_emergency_add():
    """응급 상황 등록"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'message': '요청 본문 오류'}), 400
    try:
        con.insert_data('emergency', {
            'patient_position': data.get('patient_position', ''),
            'patient_state':    _STATE_STR_TO_INT.get(data.get('patient_state', '일반'), 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True})


@app.route('/api/emergency/<int:emergency_id>', methods=['DELETE'])
def api_emergency_delete(emergency_id):
    """응급 기록 삭제"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    try:
        con.delete_data('emergency', f'emergency_id={emergency_id}')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True})


@app.route('/api/camera/status')
def api_camera_status():
    with _lock:
        available = camera_data['available']
    return ok({'available': available})


@app.route('/api/person_count')
def api_person_count():
    with person_counts_lock:
        ent = person_counts["enter"]
        ext = person_counts["exit"]
    return ok({"enter": ent, "exit": ext, "occupancy": max(0, ent - ext)})


# =============================================================================
# 순찰 실행 — plan_table DB 기반 인메모리 패트롤 리스트
# (LOCATION_DB 하드코딩 제거 → plan_table이 위치 마스터 데이터)
# =============================================================================

# 인메모리 패트롤 리스트 (ROS2 /patrol_status 미수신 시 직접 관리)
# 각 항목: {"name": str, "x": float, "y": float, "yaw": float, "detect": bool}
_patrol_list      = []
_patrol_list_lock = threading.Lock()


def _fetch_location_db() -> dict:
    """plan_table에서 위치 마스터 데이터 동적 조회.
    반환: {room_name: {"x": float, "y": float, "yaw": float, "detect": bool}}
    같은 room_name이 여러 robot에 있으면 plan_ID가 낮은(먼저 등록된) 항목 우선."""
    try:
        rows = con.select_data(table_name='plan_table', columns='*') or []
    except Exception:
        rows = []
    result = {}
    for r in rows:
        name = r.get('room_name', '')
        if name and name not in result:
            result[name] = {
                'x':      float(r.get('x_pose', 0.0)),
                'y':      float(r.get('y_pose', 0.0)),
                'yaw':    float(r.get('yaw', 0.0)),
                'detect': bool(r.get('fulfill', False)),
            }
    return result


def _loc_entry(name: str) -> dict:
    """plan_table에서 name에 해당하는 dict 반환 (없으면 기본값)"""
    db = _fetch_location_db()
    v  = db.get(name)
    if v:
        return {"name": name, "x": v['x'], "y": v['y'], "yaw": v['yaw'], "detect": v['detect']}
    return {"name": name, "x": 0.0, "y": 0.0, "yaw": 0.0, "detect": False}


def _patrol_send(payload: dict):
    """ROS2 /patrol_cmd 토픽으로 명령 전송 (ROS2 없으면 인메모리만 변경)"""
    if ros2_ready and robot_node is not None:
        try:
            msg      = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            robot_node.plan_pub.publish(msg)
        except Exception as e:
            print(f"[Patrol] ROS2 전송 실패: {e}")
    # 인메모리 동기화
    cmd = payload.get('cmd')
    with _patrol_list_lock:
        if cmd == 'set':
            _patrol_list.clear()
            for name in payload.get('list', []):
                _patrol_list.append(_loc_entry(name))
        elif cmd == 'add':
            name = payload.get('name', '')
            idx  = payload.get('index')
            if name:
                entry = _loc_entry(name)
                if idx is not None:
                    _patrol_list.insert(int(idx), entry)
                else:
                    _patrol_list.append(entry)
        elif cmd == 'remove':
            idx = payload.get('index')
            if idx is not None and 0 <= int(idx) < len(_patrol_list):
                _patrol_list.pop(int(idx))
        elif cmd == 'clear':
            _patrol_list.clear()


# ── 패트롤 페이지 ───────────────────────────────────────────────────────────
@app.route('/patrol_control')
def patrol_control():
    if 'username' not in session:
        return redirect(url_for('login'))
    # plan_table에서 위치 목록 동적 로드 (LOCATION_DB 하드코딩 제거)
    raw_plans = fetch_plan_list()
    seen_names = set()
    locations = []
    for p in raw_plans:
        name = p.get('room_name', '')
        if name and name not in seen_names:
            seen_names.add(name)
            locations.append({
                "name":   name,
                "detect": bool(p.get('fulfill', False)),
                "x":      float(p.get('x_pose', 0.0)),
                "y":      float(p.get('y_pose', 0.0)),
                "yaw":    float(p.get('yaw', 0.0)),
            })
    return render_template(
        'patrol_control.html',
        username=session['username'],
        locations=locations
    )


# ── 위치 마스터 API ─────────────────────────────────────────────────────────
@app.route('/api/locations')
def api_locations():
    """plan_table에서 위치 목록 동적 반환 (rooms DB 운영 상태 포함).
    프론트엔드에서 드롭다운 갱신 및 좌표 자동완성에 사용."""
    try:
        all_rooms = con.select_data(table_name='rooms', columns='*') or []
    except Exception:
        all_rooms = []
    active_wards  = {
        r['room_name']
        for r in all_rooms
        if r.get('existence') in (True, 1, '1', 'true')
    }
    all_ward_names = {r['room_name'] for r in all_rooms}

    def _ward_active(loc_name: str) -> bool:
        for ward in all_ward_names:
            if loc_name.startswith(ward):
                return ward in active_wards
        return True

    raw_plans   = fetch_plan_list()
    seen_names  = set()
    result      = []
    for p in raw_plans:
        name = p.get('room_name', '')
        if name and name not in seen_names:
            seen_names.add(name)
            result.append({
                'name':        name,
                'x':           float(p.get('x_pose', 0.0)),
                'y':           float(p.get('y_pose', 0.0)),
                'yaw':         float(p.get('yaw', 0.0)),
                'detect':      bool(p.get('fulfill', False)),
                'ward_active': _ward_active(name),
                'robot_num':   int(p.get('robot_num', 0)),
            })
    return jsonify({'success': True, 'locations': result})


# ── 패트롤 API (/api/patrol/...) ────────────────────────────────────────────
@app.route('/api/patrol/status')
def api_patrol_status():
    """현재 패트롤 리스트 반환 (각 항목: name, x, y, yaw, detect)"""
    with _patrol_list_lock:
        return jsonify({'success': True, 'list': list(_patrol_list)})


@app.route('/api/patrol/set', methods=['POST'])
def api_patrol_set():
    """패트롤 리스트 전체 교체"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    names = (request.get_json(silent=True) or {}).get('list', [])
    _patrol_send({'cmd': 'set', 'list': names})
    return jsonify({'success': True})


@app.route('/api/patrol/add', methods=['POST'])
def api_patrol_add():
    """위치 1개 추가"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data = request.get_json(silent=True) or {}
    name = data.get('name')
    idx  = data.get('index')
    if not name:
        return jsonify({'success': False, 'message': 'name 필요'}), 400
    payload = {'cmd': 'add', 'name': name}
    if idx is not None:
        payload['index'] = int(idx)
    _patrol_send(payload)
    return jsonify({'success': True})


@app.route('/api/patrol/remove', methods=['POST'])
def api_patrol_remove():
    """인덱스로 항목 삭제"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data = request.get_json(silent=True) or {}
    idx  = data.get('index')
    if idx is None:
        return jsonify({'success': False, 'message': 'index 필요'}), 400
    _patrol_send({'cmd': 'remove', 'index': int(idx)})
    return jsonify({'success': True})


@app.route('/api/patrol/clear', methods=['POST'])
def api_patrol_clear():
    """패트롤 리스트 전체 초기화"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    _patrol_send({'cmd': 'clear'})
    return jsonify({'success': True})


# ── plan_table ↔ patrol 연동 API ────────────────────────────────────────────

@app.route('/api/plan/list')
def api_plan_list():
    """[GET] plan_table 전체 조회 (robot_plan.html용 JSON API)"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    try:
        rows = con.select_data(table_name='plan_table', columns='*') or []
        # Mock DB는 dict 리스트, MySQL도 dict 리스트로 통일
        result = []
        for r in rows:
            result.append({
                'plan_ID':   r.get('plan_ID') or r.get('id', ''),
                'robot_num': r.get('robot_num', 1),
                'room_name': r.get('room_name', ''),
                'x_pose':    r.get('x_pose', 0.0),
                'y_pose':    r.get('y_pose', 0.0),
                'yaw':       r.get('yaw', 0.0),
                'fulfill':   bool(r.get('fulfill', False)),
            })
        return jsonify({'success': True, 'list': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/plan/run', methods=['POST'])
def api_plan_run():
    """[POST] plan_table의 특정 로봇 계획을 patrol_list로 적재 후 순찰 시작
    body: { "robot_num": 1 }  → 해당 로봇 번호의 plan_table 항목만 로드
          { "robot_num": 0 }  → 전체 로드
    """
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data      = request.get_json(silent=True) or {}
    robot_num = data.get('robot_num', 0)
    try:
        rows = con.select_data(table_name='plan_table', columns='*') or []
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

    # 필터링
    if robot_num:
        rows = [r for r in rows if int(r.get('robot_num', 0)) == int(robot_num)]

    if not rows:
        return jsonify({'success': False, 'message': '해당 계획이 없습니다.'}), 404

    # rooms DB에서 운영 중인 병동 목록 조회 (비활성 병동 제외용)
    try:
        all_rooms_run = con.select_data(table_name='rooms', columns='*') or []
    except Exception:
        all_rooms_run = []
    active_wards_run = {
        r['room_name']
        for r in all_rooms_run
        if r.get('existence') in (True, 1, '1', 'true')
    }
    all_ward_names_run = {r['room_name'] for r in all_rooms_run}

    def _is_active_run(loc_name):
        for ward in all_ward_names_run:
            if loc_name.startswith(ward):
                return ward in active_wards_run
        return True  # 소속 병동 없음 → 항상 활성

    # patrol_list 교체 (비활성 병동 위치 자동 제외)
    new_list  = []
    skip_list = []
    for r in rows:
        name = r.get('room_name', '')
        # 비활성 병동이면 건너뜀
        if not _is_active_run(name):
            skip_list.append(name)
            print(f'[plan/run] [{name}] 중단 병동 → 순찰 제외')
            continue
        # plan_table 좌표 직접 사용 (LOCATION_DB 하드코딩 제거)
        entry = {
            'name':   name,
            'x':      float(r.get('x_pose', 0.0)),
            'y':      float(r.get('y_pose', 0.0)),
            'yaw':    float(r.get('yaw', 0.0)),
            'detect': bool(r.get('fulfill', False)),
        }
        new_list.append(entry)

    # patrol_list에 직접 적재 (좌표 손실 없이)
    with _patrol_list_lock:
        _patrol_list.clear()
        _patrol_list.extend(new_list)
    # ROS2에도 전송
    if ros2_ready and robot_node is not None:
        try:
            import json as _json
            msg      = String()
            msg.data = _json.dumps({'cmd': 'set', 'list': [e['name'] for e in new_list]}, ensure_ascii=False)
            robot_node.plan_pub.publish(msg)
        except Exception as e:
            print(f'[plan/run] ROS2 전송 실패: {e}')
    print(f'[plan/run] robot_num={robot_num}, {len(new_list)}개 실행 / {len(skip_list)}개 중단 병동 제외')
    return jsonify({
        'success': True,
        'count':   len(new_list),
        'list':    new_list,
        'skipped': skip_list,
    })


@app.route('/api/plan/delete', methods=['POST'])
def api_plan_delete():
    """[POST] plan_table 항목 삭제 + patrol_list에서도 동일 room_name 제거
    body: { "plan_id": 3 }  또는  { "room_name": "room1_back" }
    """
    if 'username' not in session:
        return jsonify({'success': False}), 401
    data      = request.get_json(silent=True) or {}
    plan_id   = data.get('plan_id')
    room_name = data.get('room_name', '').strip()
    if not plan_id and not room_name:
        return jsonify({'success': False, 'message': 'plan_id 또는 room_name 필요'}), 400
    try:
        delete_plan_entry(plan_id=plan_id, room_name=room_name or None)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

    # patrol_list에서도 같은 이름 항목 제거
    if room_name:
        with _patrol_list_lock:
            before = len(_patrol_list)
            _patrol_list[:] = [e for e in _patrol_list if e.get('name') != room_name]
            removed = before - len(_patrol_list)
        if removed:
            print(f'[plan/delete] patrol_list에서 [{room_name}] {removed}개 제거')
    return jsonify({'success': True})


# =============================================================================
# 대시보드 요약 API
# =============================================================================
@app.route('/api/dashboard/summary')
def api_dashboard_summary():
    """대시보드용 rooms + patients 실시간 집계"""
    if 'username' not in session:
        return jsonify({'success': False}), 401
    try:
        all_rooms    = con.select_data(table_name='rooms',    columns='*') or []
        all_patients = con.select_data(table_name='patients', columns='*') or []
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

    # ── 병동 집계 ──────────────────────────────────────────────
    room_total  = len(all_rooms)
    room_active = sum(1 for r in all_rooms if r.get('existence') in (True, 1, '1', 'true'))

    # 병동별 상태 목록 (이름 + 활성여부)
    room_list = [
        {
            'room_id':   r.get('room_id', ''),
            'room_name': r.get('room_name', ''),
            'active':    r.get('existence') in (True, 1, '1', 'true'),
        }
        for r in all_rooms
    ]

    # 전체 병동 상태 판정
    if room_total == 0:
        ward_status = 'unknown'
        ward_label  = '정보 없음'
    elif room_active == room_total:
        ward_status = 'safe'
        ward_label  = '정상 가동'
    elif room_active == 0:
        ward_status = 'danger'
        ward_label  = '전체 중단'
    else:
        ward_status = 'warning'
        ward_label  = f'{room_active}/{room_total} 운영 중'

    # ── 환자 집계 ──────────────────────────────────────────────
    patient_total = len(all_patients)

    # 상태별 분류 (patient_state: 0=정상, 1=주의, 2=응급)
    state_map = {0: '정상', 1: '주의', 2: '응급'}
    state_count = {0: 0, 1: 0, 2: 0}
    for p in all_patients:
        s = int(p.get('patient_state', 0))
        if s in state_count:
            state_count[s] += 1

    return jsonify({
        'success': True,
        'rooms': {
            'total':   room_total,
            'active':  room_active,
            'list':    room_list,
            'status':  ward_status,   # 'safe' | 'warning' | 'danger' | 'unknown'
            'label':   ward_label,
        },
        'patients': {
            'total':   patient_total,
            'normal':  state_count[0],
            'caution': state_count[1],
            'emergency': state_count[2],
        },
    })


# =============================================================================
# 서버 진입점
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print(" 병원용 로봇 원격 제어 서버 시작")
    print(f" ROS2  : {'사용 가능' if ROS2_IMPORT_OK  else 'Mock 모드'}")
    print(f" YOLO  : {'사용 가능' if YOLO_AVAILABLE  else 'Mock 모드'}")
    print(f" cv2   : {'사용 가능' if CV2_AVAILABLE   else 'Mock 모드'}")
    print(f" MySQL : {'사용 가능' if MYSQL_AVAILABLE else 'Mock 모드'}")
    print(" http://0.0.0.0:5002")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)
