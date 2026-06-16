#!/usr/bin/env python3
"""
Person detection + pose classification + audio feedback node.

- std_msgs/Bool on 'detect_enable' opens/closes the OpenCV window
- CompressedImage input
- Publishes detected pose on 'detected_pose' (std_msgs/String)
- Publishes emergency_alert as std_msgs/String: "위급상황"
- Publishes search_done as std_msgs/String: "문제없음"
"""

import math
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from ultralytics import YOLO

from rokey_patrol.audio_manager import AudioManager
from rokey_patrol.pose_classifier import PoseClassifier
from rokey_patrol.visualizer import PersonVisualizer


WINDOW_NAME = 'Robot1 Room2 Camera'


@dataclass
class AlignerConfig:
    """
    [클래스 설명]
    이 클래스는 사람 검출/포즈 인식 노드(AmrDetect)의 모든 설정값을 한곳에 모은 데이터 클래스입니다.
    - ROS 토픽 이름(이미지/cmd_vel/이벤트 알림 등)을 보관하는 기능
    - YOLO 모델 파일과 음성 안내 파일들의 자산 경로를 보관하는 기능
    - 검출 신뢰도, 회전 속도, 정렬 마진 등 알고리즘 동작 파라미터를 보관하는 기능이 포함되어 있습니다.
    """
    image_topic:     str = 'oakd/rgb/image_raw/compressed'
    cmd_vel_topic:   str = 'cmd_vel'
    odom_topic:      str = 'odom'
    emergency_topic: str = 'emergency_alert'
    done_topic:      str = 'search_done'
    enable_topic:    str = 'detect_enable'
    pose_topic:      str = 'detected_pose'

    model_file:        str = 'yolov8s-pose.pt'
    sit_audio_file:    str = 'Please_sit.mp3'
    thanks_audio_file: str = 'thanks.mp3'
    clear_audio_file:  str = 'amr_clear.mp3'
    help_audio_file:   str = 'help_me.mp3'

    conf_threshold:          float = 0.5
    search_rotate_speed:     float = -0.3
    lost_frame_threshold:    int   = 10
    edge_margin_px:          int   = 120
    nudge_rotate_speed:      float = -0.15
    nudge_timeout_sec:       float = 0.7
    pose_confirm_frames:     int   = 5
    ask_sit_timeout_sec:     float = 10.0
    recover_confirm_frames:  int   = 3
    max_search_rotation_rad: float = field(default_factory=lambda: 2.0 * math.pi)
    rotation_tolerance_rad:  float = field(default_factory=lambda: math.radians(3.0))
    min_delta_yaw_rad:       float = field(default_factory=lambda: math.radians(0.3))
    processed_yaw_tolerance_rad: float = field(default_factory=lambda: math.radians(18.0))

    # Lying 상태에서 머리~발이 화면 안에 들어오도록 x축 미세 정렬
    lying_align_timeout_sec: float = 1.5
    lying_align_margin_px: int = 80
    lying_align_center_tol_px: int = 60
    lying_align_min_kpt_conf: float = 0.25


class AmrDetect(Node):
    """
    [클래스 설명]
    이 클래스는 카메라 영상에서 사람을 검출하고 포즈를 분류해 위급상황을 판단하는 노드입니다.
    - YOLO 포즈 모델로 사람 검출 후 Standing/Sitting/Lying 포즈를 분류하는 기능
    - 검출된 사람 위치에 따라 미세 회전(NUDGE)·전신 정렬(LYING_ALIGN)을 수행하는 기능
    - Lying 상태에서 음성으로 일어나기를 안내하고 응답이 없으면 위급상황을 알리는 기능이 포함되어 있습니다.
    """
    def __init__(self, cfg: AlignerConfig = AlignerConfig()):
        """
        ROS2 노드를 초기화하고 YOLO 모델·포즈 분류기·시각화기·오디오 매니저와 Pub/Sub 통신 채널을 설정합니다.
        [입력]: cfg (AlignerConfig, 검출/정렬 동작 파라미터 묶음)
        [출력]: None
        """
        super().__init__('amr_detect_node')
        self.cfg = cfg

        pkg_share = get_package_share_directory('rokey_patrol')
        assets = os.path.join(pkg_share, 'assets')

        model_path        = os.path.join(assets, cfg.model_file)
        sit_audio_path    = os.path.join(assets, cfg.sit_audio_file)
        thanks_audio_path = os.path.join(assets, cfg.thanks_audio_file)
        clear_audio_path  = os.path.join(assets, cfg.clear_audio_file)
        help_audio_path   = os.path.join(assets, cfg.help_audio_file)

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)

        self.pose_classifier = PoseClassifier()
        self.visualizer = PersonVisualizer(
            edge_margin_px=cfg.edge_margin_px,
            window_name=WINDOW_NAME,
        )
        self.audio = AudioManager(
            sit_audio_path=sit_audio_path,
            thanks_audio_path=thanks_audio_path,
            clear_audio_path=clear_audio_path,
            help_audio_path=help_audio_path,
            ask_sit_timeout_sec=cfg.ask_sit_timeout_sec,
            logger=self.get_logger(),
        )

        self.state = 'SEARCH'
        self.lost_count = 0
        self.is_search_rotating = False
        self.search_completed = False
        self.last_log_time = 0.0

        self.current_yaw = self.start_yaw = self.prev_yaw = None
        self.accumulated_rotation = 0.0
        self.processed_yaws: list[float] = []

        self.current_pose_label = 'Unknown'
        self.pose_history: list[str] = []
        self.recovery_history: list[str] = []

        self.nudge_start_time = None
        self.lying_align_start_time = None
        self.current_target_index = None
        self.ask_start_time = None

        self.emergency_published = False
        self.done_published = False
        self.clear_audio_played = False
        self.shutdown_requested = False

        self._enabled = False
        self._enable_lock = threading.Lock()

        self.create_subscription(CompressedImage, cfg.image_topic, self.image_callback, 10)
        self.create_subscription(Odometry, cfg.odom_topic, self.odom_callback, 10)
        self.create_subscription(Bool, cfg.enable_topic, self.on_enable, 10)

        self.cmd_pub       = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self.emergency_pub = self.create_publisher(String, cfg.emergency_topic, 10)
        self.done_pub      = self.create_publisher(String, cfg.done_topic, 10)
        self.pose_pub      = self.create_publisher(String, cfg.pose_topic, 10)

        self.get_logger().info(f'Model: {model_path}')
        self.get_logger().info(f'Image: {cfg.image_topic}  Odom: {cfg.odom_topic}')
        self.get_logger().info(f'Emergency topic: {cfg.emergency_topic} / std_msgs.msg.String')
        self.get_logger().info(f'Done topic: {cfg.done_topic} / std_msgs.msg.String')
        self.get_logger().info(f'Waiting for {cfg.enable_topic}=True ...')

    def on_enable(self, msg: Bool):
        """
        patrol_node로부터 검출 활성/비활성 신호를 수신해 OpenCV 창을 열거나 닫고 세션 상태를 초기화합니다.
        [입력]: msg (Bool, 검출 모드 활성화 트리거 신호)
        [출력]: None (내부 상태 갱신 및 OpenCV 윈도우 제어)
        """
        with self._enable_lock:
            if msg.data and not self._enabled:
                self.get_logger().info('Detection ENABLED — opening camera window.')
                self._enabled = True
                self._reset_for_new_session()

            elif not msg.data and self._enabled:
                self.get_logger().info('Detection DISABLED — closing camera window.')
                self._enabled = False
                self._halt()
                self.audio.stop_all()
                self.visualizer.close()
                try:
                    cv2.destroyWindow(WINDOW_NAME)
                except cv2.error:
                    pass

    def _reset_for_new_session(self):
        """
        새로운 검출 세션을 시작하기 위해 상태 머신·포즈 이력·이벤트 플래그 등을 모두 초기값으로 되돌립니다.
        [입력]: None
        [출력]: None
        """
        self.state = 'SEARCH'
        self.lost_count = 0
        self.is_search_rotating = False
        self.search_completed = False
        self.current_pose_label = 'Unknown'
        self.pose_history.clear()
        self.recovery_history.clear()
        self.processed_yaws.clear()
        self.accumulated_rotation = 0.0
        self.start_yaw = None
        self.prev_yaw = None
        self.nudge_start_time = None
        self.lying_align_start_time = None
        self.current_target_index = None
        self.ask_start_time = None
        self.emergency_published = False
        self.done_published = False
        self.clear_audio_played = False

    def _render(self, frame, status, mode, align, pose_text,
                people=None, pose_label=None):
        """
        시각화기에 현재 프레임과 상태 텍스트를 전달해 카메라 창을 갱신하고 ESC 입력 시 종료를 처리합니다.
        [입력]: frame (np.ndarray, 카메라 프레임), status (str, 사람 유무), mode (str, 모드 텍스트),
                align (str, 정렬 정보), pose_text (str, 포즈 표시), people (list, 검출 사람 목록), pose_label (str, 포즈 라벨)
        [출력]: None (OpenCV 윈도우 갱신)
        """
        if self.visualizer.render(
            frame,
            people=people or [],
            target_index=self.current_target_index,
            status=status,
            mode=mode,
            align=align,
            pose_text=pose_text,
            current_pose_label=pose_label or self.current_pose_label,
            start_yaw=self.start_yaw,
            current_yaw=self.current_yaw,
            processed_count=len(self.processed_yaws),
        ):
            self._on_quit()

    def _publish_string_once(self, publisher, flag_name, text, log_msg):
        """
        지정된 플래그가 False일 때만 문자열 메시지를 3회 반복 발행하고 플래그를 True로 잠급니다.
        [입력]: publisher (Publisher), flag_name (str, 발행 여부 플래그 이름),
                text (str, 발행할 문자열), log_msg (str, 로그 출력 문구)
        [출력]: None (지정 토픽 퍼블리시 및 플래그 갱신)
        """
        if getattr(self, flag_name):
            return

        msg = String()
        msg.data = text

        for _ in range(3):
            publisher.publish(msg)
            time.sleep(0.05)

        setattr(self, flag_name, True)
        self.get_logger().info(log_msg)

    def publish_emergency_once(self):
        """
        위급상황 알림('위급상황')을 emergency_alert 토픽으로 1회만 발행합니다.
        [입력]: None
        [출력]: None (emergency_alert 토픽 퍼블리시)
        """
        self._publish_string_once(
            self.emergency_pub,
            'emergency_published',
            '위급상황',
            'Emergency alert published: 위급상황'
        )

    def publish_done_once(self):
        """
        탐색 정상 완료 알림('문제없음')을 search_done 토픽으로 1회만 발행합니다.
        [입력]: None
        [출력]: None (search_done 토픽 퍼블리시)
        """
        self._publish_string_once(
            self.done_pub,
            'done_published',
            '문제없음',
            'Search done published: 문제없음'
        )

    def reset_emergency_publish_flag(self):
        """
        다음 사람을 처리할 수 있도록 위급상황 발행 잠금 플래그를 풀어줍니다.
        [입력]: None
        [출력]: None (emergency_published 플래그 False로 갱신)
        """
        self.emergency_published = False

    def odom_callback(self, msg):
        """
        오도메트리 쿼터니언에서 yaw를 추출하고 누적 회전각을 계산해 360도 탐색 완료 여부를 판단합니다.
        [입력]: msg (Odometry, 로봇의 위치·자세 메시지)
        [출력]: None (current_yaw, accumulated_rotation 등 내부 상태 갱신)
        """
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.current_yaw = yaw
        self.start_yaw = self.start_yaw or yaw

        if self.prev_yaw is None:
            self.prev_yaw = yaw
            return

        if self.state in ('SEARCH', 'NUDGE', 'LYING_ALIGN') and self.is_search_rotating and not self.search_completed:
            delta = self._angle_diff(yaw, self.prev_yaw)

            if abs(delta) >= self.cfg.min_delta_yaw_rad:
                self.accumulated_rotation += abs(delta)

                if self.accumulated_rotation >= self.cfg.max_search_rotation_rad - self.cfg.rotation_tolerance_rad:
                    self.accumulated_rotation = self.cfg.max_search_rotation_rad
                    self.search_completed = True
                    self.is_search_rotating = False
                    self.state = 'DONE'
                    self._halt()
                    self.get_logger().info('Search rotation completed: 360 degrees')

        self.prev_yaw = yaw

    @staticmethod
    def _angle_diff(a, b):
        """
        두 각도(라디안)의 차이를 -π~π 범위로 정규화하여 반환합니다.
        [입력]: a (float, 첫 번째 각도), b (float, 두 번째 각도)
        [출력]: float (정규화된 각도 차이, 라디안)
        """
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def image_callback(self, msg: CompressedImage):
        """
        압축 영상을 디코딩하고 YOLO 추론 후 현재 상태에 맞는 핸들러(SEARCH/LYING_ALIGN/ASK_TO_SIT 등)로 분기합니다.
        [입력]: msg (CompressedImage, 압축된 카메라 원본 이미지 데이터)
        [출력]: None (포즈 토픽 퍼블리시 및 상태별 핸들러 호출)
        """
        if not self._enabled:
            return

        if self.shutdown_requested:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception as e:
            self.get_logger().error(f'Image decode failed: {e}')
            return

        try:
            result = self.model(frame, verbose=False, conf=self.cfg.conf_threshold)[0]
        except Exception as e:
            self.get_logger().error(f'YOLO failed: {e}')
            return

        if self.current_yaw is None:
            self.stop_motion()
            self._render(frame, 'ODOM: WAITING', 'MODE: WAIT_ODOM',
                         'ROT: ---', 'POSE: ---', pose_label='Unknown')
            self._publish_current_pose(has_person=False, pose='Unknown')
            return

        if self.state == 'DONE':
            self._handle_done(frame, result)
            return

        if self.state == 'LYING_ALIGN':
            people = self._get_people(result, ignore_processed=False)
            self._handle_lying_align(frame, people)
            self._publish_current_pose(
                has_person=bool(people),
                pose=self.current_pose_label
            )
            return

        if self.state in ('ASK_TO_SIT', 'EMERGENCY'):
            people = self._get_people(result, ignore_processed=False)
            self._handle_lying_response(frame, people)
            self._publish_current_pose(
                has_person=bool(people),
                pose=self.current_pose_label
            )
            return

        self._handle_search(frame, result)
        self._publish_current_pose(
            has_person=self.current_target_index is not None,
            pose=self.current_pose_label
        )

    def _publish_current_pose(self, has_person: bool, pose: str):
        """
        현재 사람 유무와 포즈 라벨을 detected_pose 토픽으로 발행합니다.
        [입력]: has_person (bool, 사람 검출 여부), pose (str, 포즈 라벨)
        [출력]: None (detected_pose 토픽 퍼블리시)
        """
        msg = String()
        msg.data = pose if has_person else 'None'
        self.pose_pub.publish(msg)

    def _handle_search(self, frame, result):
        """
        SEARCH 상태에서 사람 검출 결과를 분석하여 Lying이면 정렬 모드로, 그 외엔 NUDGE 또는 포즈 안정화로 진입합니다.
        [입력]: frame (np.ndarray, 카메라 프레임), result (YOLO 결과 객체)
        [출력]: None (상태 전이 및 시각화 갱신)
        """
        people = self._get_people(result, ignore_processed=True)

        if not people:
            self._handle_no_person(frame)
            return

        self.lost_count = 0
        lying = [p for p in people if p['pose'] == 'Lying']

        if lying:
            target = max(lying, key=lambda p: p['conf'])
            self.current_pose_label = 'Lying'
            self.current_target_index = target['index']
            self._enter_lying_align()
            self._render(frame, 'PERSON: YES', 'MODE: LYING_ALIGN_START',
                         'ALIGN: checking head/foot',
                         'POSE: Lying', people=people, pose_label='Lying')
            return

        target = max(people, key=lambda p: p['conf'])
        self.current_target_index = target['index']
        self.current_pose_label = target['pose']

        person_cx = target['center_x']
        fw = frame.shape[1]
        mode_text = 'MODE: POSE_CHECK'
        align_text = f'CENTER_X: {int(person_cx)}'

        if self.is_too_close_to_edge(person_cx, fw):
            self.state = 'NUDGE'
            ready = self._nudge(person_cx, fw)
            mode_text = 'MODE: NUDGE'

            if ready:
                mode_text = self._process_stable_pose(
                    self._stable_pose(target['pose'])
                )
            else:
                self.pose_history.clear()
        else:
            self.stop_motion()
            self.state = 'SEARCH'
            self.is_search_rotating = False
            mode_text = self._process_stable_pose(
                self._stable_pose(target['pose'])
            )

        self._render(frame, 'PERSON: YES', mode_text, align_text,
                     f'POSE: {self.current_pose_label}', people=people)

    def _enter_lying_align(self):
        """
        Lying 자세 발견 시 LYING_ALIGN 상태로 전환하고 정렬 시작 시간과 오디오 상태를 초기화합니다.
        [입력]: None
        [출력]: None
        """
        self.state = 'LYING_ALIGN'
        self.is_search_rotating = False
        self.stop_motion()
        self.lying_align_start_time = time.time()
        self.audio.stop('sit')
        self.audio.stop('help')
        self.get_logger().info('Lying detected -> start lying full-body x-align')

    def _handle_lying_align(self, frame, people):
        """
        Lying 대상의 머리~발이 화면 안에 들어오도록 좌우 회전 P-control로 미세 정렬을 수행합니다.
        [입력]: frame (np.ndarray, 카메라 프레임), people (list, 검출된 사람 목록)
        [출력]: None (cmd_vel 발행 및 시각화 갱신)
        """
        lying = [p for p in people if p['pose'] == 'Lying']

        if not lying:
            self.stop_motion()
            self._enter_ask_to_sit()
            self._render(frame, 'PERSON: NO', 'MODE: LYING_ALIGN_LOST',
                         'ALIGN: lost lying -> ask',
                         'POSE: Unknown', people=people, pose_label='Unknown')
            return

        target = max(lying, key=lambda p: p['conf'])
        self.current_target_index = target['index']
        self.current_pose_label = 'Lying'

        decision, align_text = self._decide_lying_align(frame, target)

        if decision == 'DONE':
            self.stop_motion()
            self._enter_ask_to_sit()
            self._render(frame, 'PERSON: YES', 'MODE: LYING_ALIGN_DONE',
                         align_text, 'POSE: Lying',
                         people=people, pose_label='Lying')
            return

        if decision == 'GIVE_UP':
            self.stop_motion()
            self.get_logger().info('Lying align give up -> proceed with current view')
            self._enter_ask_to_sit()
            self._render(frame, 'PERSON: YES', 'MODE: LYING_ALIGN_GIVE_UP',
                         align_text, 'POSE: Lying',
                         people=people, pose_label='Lying')
            return

        twist = Twist()
        if decision == 'LEFT':
            twist.angular.z = abs(self.cfg.nudge_rotate_speed)
        elif decision == 'RIGHT':
            twist.angular.z = -abs(self.cfg.nudge_rotate_speed)

        self.cmd_pub.publish(twist)
        self.is_search_rotating = True

        self._render(frame, 'PERSON: YES', 'MODE: LYING_ALIGN',
                     align_text, 'POSE: Lying',
                     people=people, pose_label='Lying')

    def _decide_lying_align(self, frame, person):
        """
        머리·발 키포인트 또는 bbox 위치를 분석해 LEFT/RIGHT/DONE/GIVE_UP 정렬 결정을 산출합니다.
        [입력]: frame (np.ndarray, 카메라 프레임), person (dict, 검출 사람 정보)
        [출력]: tuple (str, str) — (결정 코드, 정렬 표시용 텍스트)
        """
        h, w = frame.shape[:2]
        margin = self.cfg.lying_align_margin_px
        tol = self.cfg.lying_align_center_tol_px

        if self.lying_align_start_time is None:
            self.lying_align_start_time = time.time()

        elapsed = time.time() - self.lying_align_start_time
        if elapsed >= self.cfg.lying_align_timeout_sec:
            return 'GIVE_UP', f'ALIGN: timeout {elapsed:.1f}s'

        x1, y1, x2, y2 = person['box']
        keypoints = person.get('keypoints')

        bbox_left_clipped = x1 <= 5
        bbox_right_clipped = x2 >= w - 5
        bbox_too_wide = (x2 - x1) >= w * 0.92

        if bbox_too_wide or (bbox_left_clipped and bbox_right_clipped):
            return 'GIVE_UP', 'ALIGN: too close / bbox clipped'

        xs = self._get_head_foot_xs(keypoints)

        if not xs:
            cx = (x1 + x2) / 2.0
            err = cx - (w / 2.0)

            if abs(err) <= tol:
                return 'DONE', f'ALIGN: bbox center OK err={int(err)}'

            if x1 <= margin or x2 >= w - margin:
                return 'GIVE_UP', 'ALIGN: kpt missing + bbox near edge'

            return ('RIGHT' if err > 0 else 'LEFT'), f'ALIGN: bbox err={int(err)}'

        min_x = min(xs)
        max_x = max(xs)
        body_cx = (min_x + max_x) / 2.0
        err = body_cx - (w / 2.0)

        left_near = min_x <= margin or x1 <= 5
        right_near = max_x >= w - margin or x2 >= w - 5

        if not left_near and not right_near and abs(err) <= tol:
            return 'DONE', f'ALIGN: head/foot OK err={int(err)}'

        if left_near and err < -tol:
            return 'GIVE_UP', 'ALIGN: left clipped'

        if right_near and err > tol:
            return 'GIVE_UP', 'ALIGN: right clipped'

        if abs(err) <= tol:
            return 'DONE', f'ALIGN: acceptable err={int(err)}'

        return ('RIGHT' if err > 0 else 'LEFT'), f'ALIGN: head/foot err={int(err)}'

    def _get_head_foot_xs(self, keypoints):
        """
        COCO 키포인트 중 머리(코·눈·귀)와 발(발목)의 x좌표만 신뢰도 임계값 이상인 것을 선별해 반환합니다.
        [입력]: keypoints (np.ndarray 또는 None, 17×3 키포인트 배열)
        [출력]: list (float, 신뢰도를 통과한 x좌표 목록)
        """
        if keypoints is None:
            return []

        th = self.cfg.lying_align_min_kpt_conf

        # COCO keypoint index
        # 머리: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear
        # 발: 15 left_ankle, 16 right_ankle
        indices = [0, 1, 2, 3, 4, 15, 16]
        xs = []

        for idx in indices:
            if idx >= len(keypoints):
                continue

            x, y, conf = keypoints[idx]
            if conf >= th:
                xs.append(float(x))

        return xs

    def _process_stable_pose(self, stable):
        """
        안정화된 포즈 라벨에 따라 LYING_ALIGN 진입 또는 다음 사람 처리(SEARCH 회전)를 결정합니다.
        [입력]: stable (str, 'Lying'/'Standing'/'Sitting'/'Unknown'/'Pending')
        [출력]: str (모드 표시 텍스트)
        """
        if stable == 'Lying':
            self._enter_lying_align()
            return 'MODE: LYING_ALIGN'

        if stable != 'Lying' and stable != 'Pending':
            self.mark_current_person_processed()
            self.clear_pose_tracking()
            self.reset_emergency_publish_flag()
            self.state = 'SEARCH'
            self.is_search_rotating = True
            self.rotate_to_search()
            return f'MODE: NEXT ({stable})'

        return 'MODE: POSE_CHECK'

    def _handle_no_person(self, frame):
        """
        사람이 검출되지 않을 때 lost_count를 증가시키고 임계 도달 시 SEARCH 회전을 재개합니다.
        [입력]: frame (np.ndarray, 카메라 프레임)
        [출력]: None (cmd_vel 발행 및 시각화 갱신)
        """
        self.current_pose_label = 'Unknown'
        self.current_target_index = None
        self.clear_pose_tracking()
        self.lost_count += 1

        remain_rad = self.cfg.max_search_rotation_rad - self.accumulated_rotation

        if self.search_completed:
            self._halt()
            self.state = 'DONE'
            self.is_search_rotating = False
            mode_text = 'MODE: SEARCH_DONE'
            remain_deg = 0.0

        elif self.lost_count >= self.cfg.lost_frame_threshold:
            self.state = 'SEARCH'
            self.rotate_to_search()
            self.is_search_rotating = True
            mode_text = 'MODE: SEARCH_ROTATE'
            remain_deg = max(0.0, math.degrees(remain_rad))

        else:
            self.stop_motion()
            self.is_search_rotating = False
            mode_text = 'MODE: WAITING'
            remain_deg = max(0.0, math.degrees(remain_rad))

        self._render(
            frame,
            'PERSON: NO',
            mode_text,
            f'ROT: {math.degrees(self.accumulated_rotation):.1f}/360.0  REMAIN: {remain_deg:.1f} deg',
            f'POSE: {self.current_pose_label}',
        )

    def _handle_done(self, frame, result):
        """
        360도 탐색 완료 후 정지 상태에서 search_done 알림을 1회 발행하고 클리어 음성을 재생합니다.
        [입력]: frame (np.ndarray, 카메라 프레임), result (YOLO 결과 객체)
        [출력]: None (search_done 토픽 퍼블리시 및 음성 재생)
        """
        self._halt()
        people = self._get_people(result, ignore_processed=False)

        self._render(
            frame,
            'PERSON: -',
            'MODE: SEARCH_DONE',
            'ROT: 360.0/360.0 deg  REMAIN: 0.0 deg',
            'POSE: ---',
            people=people,
            pose_label='Unknown',
        )

        if self.clear_audio_played:
            return

        self.clear_audio_played = True
        self.publish_done_once()
        cv2.waitKey(1)

        self.get_logger().info('360 scan complete -> playing clear audio')
        self.audio.play('clear', block=True)
        self._halt()

    def _enter_ask_to_sit(self):
        """
        ASK_TO_SIT 상태로 전환해 일어나기 안내 음성을 재생하고 응답 대기 타이머를 시작합니다.
        [입력]: None
        [출력]: None (오디오 재생 및 상태 전이)
        """
        self.state = 'ASK_TO_SIT'
        self.is_search_rotating = False
        self.stop_motion()
        self.ask_start_time = self.ask_start_time or time.time()
        self.lying_align_start_time = None
        self.recovery_history.clear()
        self.audio.stop('help')
        self.audio.play('sit')
        self.log_throttled('Lying detected -> sit guidance', 0.2)

    def _handle_lying_response(self, frame, people):
        """
        Lying 안내 후 사람의 반응을 관찰하고, 시간 초과 시 위급상황을, 회복 시 다음 탐색을 진행합니다.
        [입력]: frame (np.ndarray, 카메라 프레임), people (list, 검출된 사람 목록)
        [출력]: None (위급상황/회복 분기 처리 및 시각화 갱신)
        """
        self.stop_motion()

        lying = [p for p in people if p['pose'] == 'Lying']
        elapsed = 0.0 if not self.ask_start_time else time.time() - self.ask_start_time
        remain = max(0.0, self.cfg.ask_sit_timeout_sec - elapsed)
        status = 'PERSON: YES' if people else 'PERSON: NO'

        if lying:
            self.recovery_history.clear()
            self.current_pose_label = 'Lying'

            if self.state != 'EMERGENCY' and elapsed >= self.cfg.ask_sit_timeout_sec:
                self.state = 'EMERGENCY'
                self.audio.stop('sit')
                self.audio.play('help')
                self.publish_emergency_once()
                self.log_throttled('Emergency: timeout exceeded', 0.2)

            is_ask = self.state == 'ASK_TO_SIT'
            self._render(
                frame,
                status,
                'MODE: ASK_TO_SIT' if is_ask else 'MODE: EMERGENCY',
                f'REMAIN: {remain:.1f} s' if is_ask else 'REMAIN: 0.0 s',
                'POSE: Lying',
                people=people,
                pose_label='Lying',
            )
            return
        
        if self.state == 'EMERGENCY':
            self._render(
                frame,
                status,
                'MODE: EMERGENCY_HOLD',
                'WAITING FOR RESUME COMMAND...',
                'POSE: Blocked/Unknown',
                people=people,
                pose_label='Unknown',
            )
            return

        self.recovery_history.append('NotLying')

        if len(self.recovery_history) > self.cfg.recover_confirm_frames:
            self.recovery_history.pop(0)

        self.current_pose_label = 'Unknown' if not people else people[0]['pose']
        self.audio.stop('sit')
        self.audio.stop('help')

        recovered = (
            len(self.recovery_history) >= self.cfg.recover_confirm_frames
            and all(t == 'NotLying' for t in self.recovery_history)
        )

        if recovered:
            self.audio.play('thanks')
            self.mark_current_person_processed()
            self._clear_ask_state()
            self.clear_pose_tracking()
            self.reset_emergency_publish_flag()
            self.state = 'SEARCH'
            self.is_search_rotating = True
            self.rotate_to_search()

            self._render(
                frame,
                status,
                'MODE: NEXT (OK)',
                'REMAIN: 0.0 s',
                'POSE: Not Lying',
                people=people,
                pose_label='Unknown',
            )
            return

        self._render(
            frame,
            status,
            'MODE: RECOVERY_CHECK',
            f'NOT_LYING: {len(self.recovery_history)}/{self.cfg.recover_confirm_frames}',
            'POSE: Not Lying',
            people=people,
            pose_label='Unknown',
        )

    def _clear_ask_state(self):
        """
        ASK_TO_SIT 관련 타이머·이력을 초기화하고 안내·도움 음성을 모두 정지시킵니다.
        [입력]: None
        [출력]: None
        """
        self.ask_start_time = None
        self.recovery_history.clear()
        self.audio.stop('sit')
        self.audio.stop('help')

    def is_too_close_to_edge(self, cx, fw):
        """
        검출된 사람 중심점이 화면 좌우 가장자리 마진 안쪽에 들어와 있는지 판별합니다.
        [입력]: cx (float, 사람 중심 x좌표), fw (int, 프레임 가로 길이)
        [출력]: bool (가장자리에 가까우면 True)
        """
        return cx < self.cfg.edge_margin_px or cx > fw - self.cfg.edge_margin_px

    def _nudge(self, cx, fw):
        """
        사람을 화면 안쪽으로 끌어오기 위해 짧은 회전을 수행하고 안전 영역 진입 또는 타임아웃 시 종료합니다.
        [입력]: cx (float, 사람 중심 x좌표), fw (int, 프레임 가로 길이)
        [출력]: bool (안전 영역 진입 또는 타임아웃이면 True)
        """
        self.nudge_start_time = self.nudge_start_time or time.time()

        in_zone = self.cfg.edge_margin_px <= cx <= fw - self.cfg.edge_margin_px
        timed_out = (time.time() - self.nudge_start_time) >= self.cfg.nudge_timeout_sec

        if in_zone or timed_out:
            self.nudge_start_time = None
            return True

        spd = abs(self.cfg.nudge_rotate_speed)

        twist = Twist()
        twist.angular.z = -spd if self.cfg.search_rotate_speed < 0 else spd
        self.cmd_pub.publish(twist)

        self.is_search_rotating = True
        return False

    def rotate_to_search(self):
        """
        탐색 회전 속도로 cmd_vel을 발행하여 360도 탐색을 진행시키며, 이미 완료되었으면 정지시킵니다.
        [입력]: None
        [출력]: None (cmd_vel 토픽 퍼블리시)
        """
        if self.search_completed:
            self.stop_motion()
            return

        twist = Twist()
        twist.angular.z = self.cfg.search_rotate_speed
        self.cmd_pub.publish(twist)

    def stop_motion(self):
        """
        영(0) 속도 Twist를 발행해 로봇의 모든 움직임을 즉시 정지시킵니다.
        [입력]: None
        [출력]: None (cmd_vel 토픽 퍼블리시)
        """
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass

    def _halt(self):
        """
        로봇 움직임 정지와 함께 재생 중인 모든 음성을 즉시 종료합니다.
        [입력]: None
        [출력]: None (cmd_vel 정지 발행 및 오디오 정지)
        """
        self.stop_motion()
        self.audio.stop_all()

    def _stable_pose(self, pose_label):
        """
        포즈 이력 큐에 라벨을 누적하고 N프레임 동안 일관된 포즈가 관찰될 때만 안정화된 라벨을 반환합니다.
        [입력]: pose_label (str, 현재 프레임의 포즈 라벨)
        [출력]: str ('Lying'/'Standing'/'Sitting'/'Unknown'/'Pending')
        """
        self.pose_history.append(pose_label)

        if len(self.pose_history) > self.cfg.pose_confirm_frames:
            self.pose_history.pop(0)

        if len(self.pose_history) < self.cfg.pose_confirm_frames:
            return 'Pending'

        if all(p == 'Lying' for p in self.pose_history):
            return 'Lying'

        if all(p in ('Standing', 'Sitting', 'Unknown') for p in self.pose_history):
            return self.pose_history[-1]

        return 'Pending'

    def clear_pose_tracking(self):
        """
        포즈 안정화 이력과 NUDGE/LYING_ALIGN 타이머를 초기화하여 추적 상태를 리셋합니다.
        [입력]: None
        [출력]: None
        """
        self.pose_history.clear()
        self.nudge_start_time = None
        self.lying_align_start_time = None

    def _is_yaw_processed(self, yaw):
        """
        주어진 yaw가 이미 처리한 yaw 목록 중 하나와 허용 오차 이내로 겹치는지 확인합니다.
        [입력]: yaw (float, 검사 대상 yaw 라디안)
        [출력]: bool (이미 처리된 yaw이면 True)
        """
        tol = self.cfg.processed_yaw_tolerance_rad
        return any(abs(self._angle_diff(yaw, y)) <= tol for y in self.processed_yaws)

    def mark_current_person_processed(self):
        """
        현재 yaw를 처리 완료 목록에 추가하여 같은 사람이 다시 검출되지 않도록 막습니다.
        [입력]: None
        [출력]: None (processed_yaws 목록 갱신)
        """
        if self.current_yaw is None or self._is_yaw_processed(self.current_yaw):
            return

        self.processed_yaws.append(self.current_yaw)
        self.get_logger().info(
            f'Processed yaw: {math.degrees(self.current_yaw):.1f} deg '
            f'(total={len(self.processed_yaws)})'
        )

    def is_current_yaw_processed(self):
        """
        현재 로봇이 바라보는 yaw가 이미 처리 완료 목록에 들어 있는지 확인합니다.
        [입력]: None
        [출력]: bool (이미 처리된 yaw이면 True)
        """
        return self.current_yaw is not None and self._is_yaw_processed(self.current_yaw)

    def _get_people(self, result, ignore_processed=True):
        """
        YOLO 결과에서 사람 클래스만 골라 신뢰도/포즈/키포인트를 채워 사람 정보 딕셔너리 리스트로 반환합니다.
        [입력]: result (YOLO 결과 객체), ignore_processed (bool, 이미 처리된 yaw면 빈 리스트 반환 여부)
        [출력]: list (사람 정보 dict 목록 — index, conf, box, center_x, pose, keypoints)
        """
        if result.boxes is None or not len(result.boxes):
            return []

        if ignore_processed and self.is_current_yaw_processed():
            return []

        people = []

        try:
            boxes = result.boxes

            for i in range(len(boxes)):
                if int(boxes.cls[i].item()) != 0:
                    continue

                conf = float(boxes.conf[i].item())

                if conf < self.cfg.conf_threshold:
                    continue

                x1, y1, x2, y2 = (float(v.item()) for v in boxes.xyxy[i])
                pose = 'Unknown'
                keypoints = None

                if result.keypoints is not None and i < len(result.keypoints.data):
                    keypoints = result.keypoints.data[i].cpu().numpy()

                    if hasattr(self.pose_classifier, 'classify'):
                        pose = self.pose_classifier.classify(
                            keypoints,
                            np.array([x1, y1, x2, y2], dtype=np.float32)
                        )
                    else:
                        pose = self.pose_classifier.classify_raw(
                            keypoints,
                            np.array([x1, y1, x2, y2], dtype=np.float32)
                        )

                people.append({
                    'index': i,
                    'conf': conf,
                    'box': (x1, y1, x2, y2),
                    'center_x': (x1 + x2) / 2.0,
                    'pose': pose,
                    'keypoints': keypoints,
                })

        except Exception as e:
            self.get_logger().warn(f'People parse error: {e}')

        return people

    def log_throttled(self, msg, interval=1.0):
        """
        지정 시간 간격 이상 지나야 로그를 출력하도록 제한하여 로그 폭주를 방지합니다.
        [입력]: msg (str, 출력할 로그 문자열), interval (float, 최소 간격 초)
        [출력]: None (조건 만족 시 로그 출력)
        """
        now = time.time()

        if now - self.last_log_time >= interval:
            self.get_logger().info(msg)
            self.last_log_time = now

    def _on_quit(self):
        """
        ESC 키 입력 등 종료 신호 발생 시 로봇 정지·시각화 종료·rclpy 셧다운을 수행합니다.
        [입력]: None
        [출력]: None (rclpy 셧다운)
        """
        self.get_logger().info('ESC -> shutdown')
        self._halt()
        self.visualizer.close()
        self.shutdown_requested = True
        rclpy.shutdown()

    def destroy_node(self):
        """
        노드 종료 직전 로봇 정지, 시각화 창 닫기, OpenCV 자원 해제를 수행한 뒤 부모 destroy를 호출합니다.
        [입력]: None
        [출력]: None
        """
        self._halt()
        self.visualizer.close()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        super().destroy_node()


def main(args=None):
    """
    스크립트 실행 시 ROS2 환경을 구성하고 AmrDetect 노드를 가동하며 종료 시 자원을 정리합니다.
    [입력]: args (list, 명령줄 인수)
    [출력]: None
    """
    rclpy.init(args=args)
    node = AmrDetect()

    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node._halt()
        node.visualizer.close()
        node.destroy_node()

        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
