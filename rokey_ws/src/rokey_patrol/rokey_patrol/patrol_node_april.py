#!/usr/bin/env python3
import math
import time
import threading
import json

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String
from sensor_msgs.msg import BatteryState

# x, y, yaw, detect_on_arrival
LOCATION_DB = {
    "info_front":        (-1.843,  0.098,  3.059, False),
    "info_hall_front":   (-2.708,  0.149,  3.060, False),
    "info_hall_left":    (-2.785,  0.128, -1.624, False),
    "info_hall_right":   (-2.814,  0.081,  1.900, False),
    "room1_hall_left":   (-2.924, -1.475, -1.644, False),
    "room1_back":        (-2.093, -1.596, -0.058, True),
    "room1_in_front":    (-0.348, -1.733, 3.076, False),
    "room2_hall_right":  (-2.769,  1.400,  1.421, False),
    "room2_hall_back":   (-2.767,  1.448, -0.121, False),
    "room2_back":        (-1.774,  1.585, -0.116, True),
    "room2_in_fr_back":  (-0.181,  1.357,  3.113, False),
    "room2_in_bed_left": (-1.085,  3.497, -1.763, False),
    "hall_end_left":     (-2.354,  3.890, -1.781, False),
}

def yaw_to_quat(yaw):
    """
    헤딩 각도(yaw)를 ROS 쿼터니언 4원소 튜플(x, y, z, w)로 변환합니다.
    [입력]: yaw (float, 라디안 단위 회전각)
    [출력]: tuple (float, float, float, float) — (qx, qy, qz, qw) 쿼터니언
    """
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PatrolNode(Node):
    """
    [클래스 설명]
    이 클래스는 병원 병동 로봇의 웹 명령 기반 순찰 흐름과 초기화 시퀀스를 총괄합니다.
    - 도킹 상태 확인 후 언도킹·전진을 수행하고 AprilTag 영점 조정을 요청하는 초기 시퀀스 기능
    - 웹에서 들어온 JSON 명령(set/add/remove/clear/resume)으로 순찰 리스트를 동적 관리하는 기능
    - Nav2 액션을 통한 웨이포인트 주행 및 위급상황 발생 시 정지·재개 제어 기능이 포함되어 있습니다.
    """
    def __init__(self):
        """
        ROS2 노드를 초기화하고 콜백 그룹, Nav2/카메라 클라이언트, Pub/Sub 통신 채널, 순찰 스레드를 설정합니다.
        [입력]: None
        [출력]: None
        """
        super().__init__('patrol_node')
        self.cb_group = ReentrantCallbackGroup()

        self.patrol_list      = []
        self.list_lock        = threading.Lock()
        self.cancel_requested = False

        self.detect_result = None
        self.detect_lock   = threading.Lock()
        self.detect_event  = threading.Event()

        # --- 추가: 초기화 시퀀스용 상태 변수 ---
        self.is_docked = None
        self.loc_ready_event = threading.Event()

        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.cb_group
        )

        self.camera_client = self.create_client(
            Trigger, '/robot1/oakd/start_camera',
            callback_group=self.cb_group
        )

        # Publishers
        self.cmd_pub       = self.create_publisher(Twist, 'cmd_vel', 10)  # 추가: 로봇 수동 제어용
        self.loc_start_pub = self.create_publisher(Bool, '/loc_start', 10) # 추가: 영점 조절 시작 신호
        self.detect_pub    = self.create_publisher(Bool, 'detect_enable',  10)
        self.status_pub    = self.create_publisher(String, '/patrol_status',  10)

        # Subscribers
        self.create_subscription(
            BatteryState, 'battery_state', self.battery_callback, 10,
            callback_group=self.cb_group
        ) # 추가: 배터리(도킹) 상태 구독
        self.create_subscription(
            Bool, '/loc_ready', self.on_loc_ready, 10,
            callback_group=self.cb_group
        ) # 추가: 영점 조절 완료 신호 구독
        self.create_subscription(
            String, '/patrol_cmd', self.cmd_callback, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            String, 'emergency_alert', self.on_emergency, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            String, 'search_done', self.on_search_done, 10,
            callback_group=self.cb_group
        )

        self.create_timer(1.0, self.publish_status, callback_group=self.cb_group)

        self._start_camera()
        self._set_detect(False)

        self.patrol_thread = threading.Thread(target=self.patrol_loop, daemon=True)
        self.patrol_thread.start()

        self.get_logger().info('Patrol node ready. Waiting for waypoints...')
        self.get_logger().info(f'Known locations: {list(LOCATION_DB.keys())}')

    # =========================================================
    # 추가된 콜백 (Battery & Localization)
    # =========================================================
    def battery_callback(self, msg: BatteryState):
        """
        배터리 상태 메시지의 power_supply_status 값으로 현재 도킹 여부를 판별합니다.
        [입력]: msg (BatteryState, 배터리 충전/전원 상태 정보)
        [출력]: None (self.is_docked 플래그 갱신)
        """
        # 1: Charging, 4: Full -> 보통 도킹 상태를 의미함
        self.is_docked = (msg.power_supply_status in [1, 4])

    def on_loc_ready(self, msg: Bool):
        """
        localization 노드로부터 영점 조정 완료 신호를 수신해 대기 이벤트를 해제합니다.
        [입력]: msg (Bool, 영점 조정 완료 여부)
        [출력]: None (loc_ready_event 셋)
        """
        if msg.data:
            self.loc_ready_event.set()

    # =========================================================
    # Camera & Detection
    # =========================================================
    def _start_camera(self):
        """
        OAK-D 카메라 노드의 start_camera 서비스를 호출해 영상 스트림을 활성화합니다.
        [입력]: None
        [출력]: None
        """
        if not self.camera_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('start_camera service not available, skipping.')
            return
        future = self.camera_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() is not None:
            self.get_logger().info(f'Camera started: {future.result().message}')
        else:
            self.get_logger().warn('Camera start call failed.')

    def _set_detect(self, enable: bool):
        """
        amr_detect 노드에 사람 검출 모드의 활성/비활성 상태를 토픽으로 전달합니다.
        [입력]: enable (bool, 검출 활성화 여부)
        [출력]: None (detect_enable 토픽 퍼블리시)
        """
        msg = Bool()
        msg.data = bool(enable)
        self.detect_pub.publish(msg)
        self.get_logger().info(f'detect_enable → {enable}')

    def on_emergency(self, msg: String):
        """
        amr_detect로부터 위급상황 알림을 수신해 검출 결과를 'emergency'로 기록합니다.
        [입력]: msg (String, '위급상황' 문자열을 담은 알림 메시지)
        [출력]: None (detect_result 갱신 및 detect_event 셋)
        """
        if msg.data == '위급상황':
            self.get_logger().warn('EMERGENCY received!')
            with self.detect_lock:
                self.detect_result = 'emergency'
            self.detect_event.set()

    def on_search_done(self, msg: String):
        """
        amr_detect로부터 탐색 정상 완료 신호를 수신해 검출 결과를 'done'으로 기록합니다.
        [입력]: msg (String, '문제없음' 문자열을 담은 완료 메시지)
        [출력]: None (detect_result 갱신 및 detect_event 셋)
        """
        if msg.data == '문제없음':
            self.get_logger().info('Search done — no problem.')
            with self.detect_lock:
                self.detect_result = 'done'
            self.detect_event.set()

    def _wait_for_detect_result(self):
        """
        검출 모드를 켠 후 amr_detect의 결과(emergency 또는 done)가 올 때까지 대기합니다.
        [입력]: None
        [출력]: str ('emergency', 'done', 또는 None — 취소 시)
        """
        with self.detect_lock:
            self.detect_result = None
        self.detect_event.clear()

        self._set_detect(True)
        self.get_logger().info('Waiting for amr_detect result...')

        while not self.detect_event.wait(timeout=1.0):
            with self.list_lock:
                if self.cancel_requested:
                    break

        with self.detect_lock:
            result = self.detect_result

        if result != 'emergency':
            self._set_detect(False)

        self.get_logger().info(f'Detect result: {result}')
        return result

    # =========================================================
    # 추가된 우선 수행 시퀀스 (언도킹 -> 전진 -> 영점조정)
    # =========================================================
    def _perform_initial_undock_and_move(self):
        """
        도킹 상태를 확인해 도킹된 경우 후진 언도킹을 수행한 뒤 일정 시간 전진하여 안전 위치로 이동합니다.
        [입력]: None
        [출력]: None (cmd_vel 토픽 퍼블리시)
        """
        self.get_logger().info('배터리/도킹 상태 정보를 기다리는 중...')
        
        while self.is_docked is None and rclpy.ok():
            time.sleep(0.1)

        if self.is_docked:
            self.get_logger().info('🟢 도킹 상태 감지됨. 언도킹을 시작합니다 (후진).')
            msg = Twist()
            msg.linear.x = -0.15
            self.cmd_pub.publish(msg)
            time.sleep(1.5)
            self.cmd_pub.publish(Twist()) # 정지
            time.sleep(0.5)
        else:
            self.get_logger().info('⚪ 이미 언도킹 상태입니다. 후진을 생략합니다.')

        self.get_logger().info('▶️ 앞으로 2초간 전진합니다.')
        msg = Twist()
        msg.linear.x = 0.2
        self.cmd_pub.publish(msg)
        time.sleep(5.0)
        self.cmd_pub.publish(Twist()) # 정지
        time.sleep(0.5)

    def _wait_for_loc_ready(self):
        """
        loc_start 신호를 발행해 영점 재조정을 요청하고 완료 신호가 올 때까지 최대 20초간 대기합니다.
        [입력]: None
        [출력]: None (loc_start 토픽 퍼블리시)
        """
        self.loc_ready_event.clear()
        
        msg = Bool()
        msg.data = True
        self.loc_start_pub.publish(msg)
        self.get_logger().info('🎯 영점 재조정(AprilTag Localization) 시작. 대기 중...')

        # 타임아웃 20초 (enji_auto_localization.py 내부에도 타임아웃이 존재함)
        if self.loc_ready_event.wait(timeout=20.0):
            self.get_logger().info('✅ 영점 재조정 완료! 정확한 위치를 파악했습니다.')
        else:
            self.get_logger().warn('⚠️ 영점 재조정 응답 시간 초과 (Timeout).')

    # =========================================================
    # Web cmd
    # =========================================================
    def cmd_callback(self, msg: String):
        """
        웹 인터페이스에서 보낸 JSON 명령을 파싱하여 순찰 리스트를 set/add/remove/clear/resume 합니다.
        [입력]: msg (String, JSON 형식의 순찰 제어 명령)
        [출력]: None (patrol_list 갱신 및 상태 플래그 변경)
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'Invalid JSON: {msg.data}')
            return

        cmd = data.get('cmd')
        with self.list_lock:
            if cmd == 'set':
                self.patrol_list = list(data.get('list', []))
                self.cancel_requested = True
                self.get_logger().info(f'SET → {self.patrol_list}')
            elif cmd == 'add':
                name = data.get('name')
                idx  = data.get('index')
                if name not in LOCATION_DB:
                    self.get_logger().warn(f'Unknown location: {name}')
                    return
                if idx is None:
                    self.patrol_list.append(name)
                else:
                    self.patrol_list.insert(int(idx), name)
                self.get_logger().info(f'ADD → {self.patrol_list}')
            elif cmd == 'remove':
                idx = data.get('index')
                if idx is not None and 0 <= int(idx) < len(self.patrol_list):
                    removed = self.patrol_list.pop(int(idx))
                    self.get_logger().info(f'REMOVE {removed} → {self.patrol_list}')
            elif cmd == 'clear':
                self.patrol_list.clear()
                self.cancel_requested = True
                self.detect_event.set()
                self.get_logger().info('CLEAR')
            elif cmd == 'resume':
                with self.detect_lock:
                    if self.detect_result == 'emergency':
                        self.detect_result = 'done'
                self.get_logger().info('RESUME → Emergency cleared, resuming patrol.')

    def publish_status(self):
        """
        현재 순찰 리스트 상태를 1초마다 JSON 문자열로 직렬화해 웹 인터페이스에 방송합니다.
        [입력]: None
        [출력]: None (patrol_status 토픽 퍼블리시)
        """
        msg = String()
        with self.list_lock:
            msg.data = json.dumps({'list': self.patrol_list}, ensure_ascii=False)
        self.status_pub.publish(msg)

    # =========================================================
    # Patrol loop
    # =========================================================
    def patrol_loop(self):
        """
        초기 언도킹·영점조정 시퀀스를 1회 수행한 뒤 순찰 리스트를 순회하며 주행/검출 흐름을 실행합니다.
        [입력]: None
        [출력]: None (Nav2 주행 명령 및 검출 트리거 발행)
        """
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 action server ready.')

        # ---------------------------------------------------------
        # [우선 수행 시퀀스] 순찰 리스트를 처리하기 전에 1회 무조건 실행됩니다.
        # ---------------------------------------------------------
        self._perform_initial_undock_and_move()
        self._wait_for_loc_ready()
        self.get_logger().info('🚀 필수 초기화 시퀀스 종료. 웹 명령(순찰 리스트) 처리 대기 상태 진입.')
        # ---------------------------------------------------------

        while rclpy.ok():
            with self.list_lock:
                snapshot = list(self.patrol_list)

            if not snapshot:
                time.sleep(0.5)
                continue

            for name in snapshot:
                with self.list_lock:
                    if name not in self.patrol_list:
                        break
                    if self.cancel_requested:
                        self.cancel_requested = False
                        break

                if name not in LOCATION_DB:
                    self.get_logger().warn(f'Skip unknown: {name}')
                    continue

                x, y, yaw, do_detect = LOCATION_DB[name]
                self.get_logger().info(
                    f'→ [{name}]  x={x} y={y} yaw={yaw:.3f}  detect={do_detect}'
                )

                success = self.send_goal_and_wait(x, y, yaw)

                if not success:
                    self.get_logger().warn(f'Navigation to [{name}] failed, moving on.')
                    continue

                if do_detect:
                    result = self._wait_for_detect_result()
                    if result == 'emergency':
                        self.get_logger().warn(f'EMERGENCY at [{name}] — holding position.')
                        # hold until search_done or list cleared
                        while rclpy.ok():
                            with self.list_lock:
                                if self.cancel_requested:
                                    break
                            with self.detect_lock:
                                if self.detect_result == 'done':
                                    break
                            time.sleep(0.5)
                            
                        self._set_detect(False)

    # =========================================================
    # Nav2
    # =========================================================
    def send_goal_and_wait(self, x, y, yaw) -> bool:
        """
        Nav2 액션 서버에 목표 좌표를 전송하고 결과가 나올 때까지 폴링하며 취소 요청도 처리합니다.
        [입력]: x (float, x좌표), y (float, y좌표), yaw (float, 헤딩각)
        [출력]: bool (정상 도착하면 True, 거절·취소·실패 시 False)
        """
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quat(float(yaw))
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        send_future = self.nav_client.send_goal_async(goal)
        while not send_future.done() and rclpy.ok():
            time.sleep(0.1)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done() and rclpy.ok():
            with self.list_lock:
                if self.cancel_requested:
                    goal_handle.cancel_goal_async()
                    self.get_logger().info('Goal canceled — list changed.')
                    return False
            time.sleep(0.1)

        status = result_future.result().status
        if status == 4:
            self.get_logger().info('Goal reached.')
            return True
        else:
            self.get_logger().warn(f'Goal ended with status={status}.')
            return False


def main():
    """
    스크립트 실행 시 ROS2 환경을 구성하고 MultiThreadedExecutor로 순찰 노드를 가동합니다.
    [입력]: None
    [출력]: None
    """
    rclpy.init()
    node = PatrolNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
