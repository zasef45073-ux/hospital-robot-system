"""
[노드명] LocalizationNode (localization_node)
[역할]   /loc_start 신호를 받으면 AprilTag 5샘플 수집 후 평균 위치를 /initialpose로 발행, 완료 시 /loc_ready 발행

[구독]   /loc_start (std_msgs/Bool)                                  — patrol_node 로컬라이제이션 요청
[구독]   oakd/rgb/camera_info (sensor_msgs/CameraInfo)               — 카메라 내부 파라미터
[구독]   oakd/rgb/image_raw/compressed (sensor_msgs/CompressedImage) — 압축 RGB 영상
[발행]   /initialpose (geometry_msgs/PoseWithCovarianceStamped)      — AMCL 초기 위치
[발행]   /loc_ready (std_msgs/Bool)                                  — patrol_node 완료 신호
[발행]   /cmd_vel (geometry_msgs/Twist)                              — 로봇 정렬 회전/전진 제어

[동작 흐름]  (★ tag_localization.py 방식: 정렬과 접근을 동시 P-control)
  1. /loc_start True  → count/samples 리셋, active=True
  2. 이미지 콜백에서 AprilTag 검출 시:
     - set_calc_align으로 tx·yaw·distance 계산
     - set_align: yaw·tx·distance를 동시에 P-control (분리하지 않음)
     - 셋 다 임계값 이내면 자동 정지 → 샘플 수집
  3. 5샘플 도달  → 평균 pose 계산 → /initialpose 발행 → /loc_ready True 발행
  4. 60초 타임아웃 → 경고 로그 → /loc_ready True 발행 (태그 미검출 상황)
"""
import time
from typing import Any, List, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from pupil_apriltags import Detector, Detection
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Bool
from tf_transformations import quaternion_from_matrix, quaternion_matrix


class LocalizationNode(Node):
    """
    [클래스 설명]
    이 클래스는 AprilTag 비전 기반 AMCL 초기 위치 추정 및 정렬 주행을 담당합니다.
    - 카메라 영상에서 AprilTag를 검출해 tx·yaw·거리 오차를 계산하는 비전 처리 기능
    - yaw·tx·거리를 동시에 P-control하여 태그 정면 1m 지점으로 정렬 주행하는 기능
    - 정렬 완료 후 5샘플의 평균 pose를 산출해 /initialpose로 AMCL을 보정하는 기능이 포함되어 있습니다.
    """

    def __init__(self) -> None:
        """
        ROS2 노드를 초기화하고 AprilTag 검출기, 카메라/제어 파라미터, Pub/Sub 통신 채널, 타임아웃 타이머를 설정합니다.
        [입력]: None
        [출력]: None
        """
        super().__init__('localization_node')

        # AprilTag 검출기 초기화
        self.detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.5,
            quad_sigma=0.5,
            refine_edges=1,
            decode_sharpening=1,
            debug=0,
        )

        # 카메라 내부 파라미터 (camera_info_callback에서 1회 채워짐)
        self.camera_params: Optional[tuple] = None
        self.K: Optional[np.ndarray] = None
        self.tag_size = 0.11  # AprilTag 한 변 실제 길이 (미터)

        # 실측 거리 보정 스케일 (측정값 × scale = 보정값) — apriltag_info.py 기준
        self.dist_scale = 1.0 / 0.582
        # 목표 거리 (보정 거리 기준, 미터) — 태그 앞 1.0m에서 정렬
        self.target_distance = 1.0
        # 샘플 수집 거리 상한 (보정 거리 기준, 미터)
        self.max_sample_dist = 1.1  # target + 여유

        # 정렬 임계값
        self.align_yaw_th = 5.0       # 도
        self.align_tx_th = 0.05       # 미터
        self.align_dist_th = 0.05     # 미터 (target_distance 대비)

        # tag_id별 map 기준 태그 위치 행렬 — 실제 환경에 맞게 설정 필요
        self.tag_poses: dict[int, np.ndarray] = {
            # 예시: 0: np.array([[1,0,0,2.0],[0,1,0,1.0],[0,0,1,0],[0,0,0,1]], dtype=float),
        }
        # 셋팅필요
        self.default_map_T_tag: np.ndarray = np.eye(4)
        self.default_map_T_tag[0, 3] = 2.0
        self.default_map_T_tag[1, 3] = 1.0

        self.T_camera_to_robot: np.ndarray = np.eye(4)  # camera → robot 고정 TF
        self.T_camera_to_robot[2, 3] = -0.2

        # 샘플 수집 상태
        self.active = False
        self.start_time = 0.0
        self.timeout_sec = 60.0
        self.sample_goal = 5
        self.samples: List[np.ndarray] = []  # T_map_to_robot 행렬 목록

        # Publishers
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )
        self.loc_ready_pub = self.create_publisher(Bool, '/loc_ready', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.create_subscription(Bool, '/loc_start', self.loc_start_callback, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'oakd/rgb/camera_info',
            self.camera_info_callback,
            10,
        )
        self.create_subscription(
            CompressedImage,
            'oakd/rgb/image_raw/compressed',
            self.image_callback,
            10,
        )

        # 타임아웃 감시 타이머 (1Hz)
        self.create_timer(1.0, self.check_timeout)

        self.get_logger().info('Localization node started.')

    # =========================================================
    # Callbacks
    # =========================================================
    def loc_start_callback(self, msg: Bool) -> None:
        """
        patrol_node로부터 영점 조정 요청 신호를 수신해 샘플 수집 모드를 활성화합니다.
        [입력]: msg (Bool, 영점 조정 시작 트리거 신호)
        [출력]: None (active 플래그 및 샘플 버퍼 초기화)
        """
        if not msg.data:
            return
        self.active = True
        self.start_time = time.time()
        self.samples = []
        self.get_logger().info('/loc_start 수신 — 샘플 수집 시작')

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        OAK-D 카메라의 내부 파라미터(K 행렬, fx/fy/cx/cy)를 1회 수신해 저장하고 이후 구독을 해제합니다.
        [입력]: msg (CameraInfo, 카메라 렌즈 보정 및 파라미터 정보)
        [출력]: None (camera_params 갱신)
        """
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            self.camera_params = (fx, fy, cx, cy)
            self.get_logger().info(
                f'카메라 파라미터 수신: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}'
            )
        else:
            self.destroy_subscription(self.camera_info_sub)

    def set_calc_align(self, results: List[Detection]):
        """
        검출된 AprilTag 중 가장 가까운 태그를 골라 tx·yaw·거리 오차와 정렬 완료 여부를 계산합니다.
        [입력]: results (List[Detection], pupil_apriltags가 검출한 태그 리스트)
        [출력]: tuple (det, aligned, tx, yaw, distance) 또는 None — 태그 없으면 None
        """
        if not results:
            return None
        det: Detection = min(results, key=lambda d: float(np.linalg.norm(d.pose_t)))
        R_tc = det.pose_R.T
        t_tc = (-R_tc @ det.pose_t).flatten()
        tx = float(t_tc[0])
        yaw = float(np.degrees(np.arctan2(R_tc[0, 2], R_tc[2, 2])))
        distance = self.correct_dist(float(np.linalg.norm(det.pose_t)))

        # 정렬 완료 조건: 회전 + 좌우 오프셋 + 거리 모두 임계값 이내
        error_dist = distance - self.target_distance
        aligned = (
            abs(yaw) < self.align_yaw_th
            and abs(tx) < self.align_tx_th
            and abs(error_dist) < self.align_dist_th
        )

        self.get_logger().info(
            f'[Tag {det.tag_id}] dist={distance:.3f}m  tx={tx:.3f}m  '
            f'yaw={yaw:.2f}°  err_dist={error_dist:+.3f}m  aligned={aligned}'
        )
        return det, aligned, tx, yaw, distance

    def set_align(self, align_info) -> None:
        """
        yaw·tx·거리 오차에 비례한 회전/전진 속도를 동시 P-control로 산출해 cmd_vel로 발행합니다.
        [입력]: align_info (tuple 또는 None, set_calc_align의 반환값)
        [출력]: None (cmd_vel 토픽 퍼블리시)
        """
        twist = Twist()

        # 태그를 놓쳤으면 정지
        if align_info is None:
            self.cmd_vel_pub.publish(twist)
            return

        det, aligned, tx, yaw, distance = align_info

        # 정렬 완료 → 정지 (샘플 수집 단계로 진입)
        if aligned:
            self.cmd_vel_pub.publish(twist)
            return

        # ---------- 동시 P-control ----------
        # 제어 계수
        KP_YAW  = 0.03    # 도 → rad/s
        KP_TX   = 0.8     # m   → rad/s (yaw가 작을 때 미세조정)
        KP_DIST = 0.3     # m   → m/s

        # 속도 제한
        MAX_ANG = 0.3
        MIN_ANG = 0.15    # 바퀴가 구를 수 있는 최소 회전 출력
        MAX_LIN = 0.1
        MIN_LIN = 0.03    # 바퀴가 구를 수 있는 최소 전진 출력

        # 1. 회전 속도 계산
        if abs(yaw) >= self.align_yaw_th:
            ang_vel = KP_YAW * yaw
        else:
            ang_vel = KP_TX * tx

        if abs(ang_vel) > 1e-4:
            sign = 1.0 if ang_vel >= 0 else -1.0
            twist.angular.z = float(sign * max(MIN_ANG, min(abs(ang_vel), MAX_ANG)))
        else:
            twist.angular.z = 0.0

        # 2. 전진/후진 속도 계산 — 거리 오차에 비례
        error_dist = distance - self.target_distance
        if abs(error_dist) >= self.align_dist_th:
            lin_vel = KP_DIST * error_dist
            sign = 1.0 if lin_vel >= 0 else -1.0
            twist.linear.x = float(sign * max(MIN_LIN, min(abs(lin_vel), MAX_LIN)))
        else:
            twist.linear.x = 0.0

        self.cmd_vel_pub.publish(twist)

    def image_callback(self, msg: CompressedImage) -> None:
        """
        매 프레임마다 압축 영상을 디코딩해 AprilTag를 검출하고, 정렬 P-control 후 정렬 완료 시 샘플을 누적합니다.
        [입력]: msg (CompressedImage, 압축된 카메라 원본 이미지 데이터)
        [출력]: None (cmd_vel 발행, 샘플 버퍼 누적, OpenCV 시각화)
        """
        if not self.active or self.camera_params is None:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )
        display_frame = frame.copy()

        # 화면에 보이는 모든 AprilTag에 초록색 박스와 ID 그리기
        for r in results:
            corners = np.int32(r.corners)
            cv2.polylines(display_frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cx, cy = int(r.center[0]), int(r.center[1])
            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"ID: {r.tag_id}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # PC 화면에 창 띄우기
        cv2.putText(display_frame, "Searching AprilTag...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.imshow("AprilTag Live Cam", display_frame)
        cv2.waitKey(1)

        if not results:
            return

        align_info = self.set_calc_align(results)
        if align_info is None:
            return

        det, aligned, _, _, distance = align_info

        # 항상 제어 (정렬 완료 시 set_align 내부에서 자동 정지)
        self.set_align(align_info)

        # 정렬 미완료면 샘플 수집 안 함
        if not aligned:
            return

        # 정렬 완료 후 거리 안전 체크
        if distance >= self.max_sample_dist:
            self.get_logger().info(
                f'거리 초과로 샘플 무시 [tag_id={det.tag_id}, dist={distance:.3f}m]'
            )
            return

        T_map_to_robot = self.compute_pose(det.tag_id, det.pose_R, det.pose_t)
        self.samples.append(T_map_to_robot)
        self.get_logger().info(
            f'샘플 수집 {len(self.samples)}/{self.sample_goal} '
            f'[tag_id={det.tag_id}, dist={distance:.3f}m]'
        )

        if len(self.samples) >= self.sample_goal:
            self._finish_localization()

    def check_timeout(self) -> None:
        """
        1Hz로 활성 시간을 감시하여 60초가 초과되면 경고와 함께 /loc_ready를 강제로 발행합니다.
        [입력]: None
        [출력]: None (loc_ready 토픽 퍼블리시 — 타임아웃 시)
        """
        if not self.active:
            return
        elapsed = time.time() - self.start_time
        if elapsed >= self.timeout_sec:
            self.get_logger().warn(
                f'로컬라이제이션 타임아웃 ({self.timeout_sec:.0f}s) — '
                '태그 미검출, 순찰 재개'
            )
            self._publish_ready()

    # =========================================================
    # Localization
    # =========================================================
    def _finish_localization(self) -> None:
        """
        수집된 5개 샘플의 평균 pose를 계산해 /initialpose로 발행하고 완료 처리를 진행합니다.
        [입력]: None
        [출력]: None (initialpose, loc_ready 토픽 퍼블리시)
        """
        T_avg = self._average_transforms(self.samples)
        msg = self.to_msg(T_avg)
        self.initialpose_pub.publish(msg)
        self.get_logger().info(
            f'/initialpose 발행 완료 — '
            f'x={T_avg[0, 3]:.3f} y={T_avg[1, 3]:.3f}'
        )
        self._publish_ready()

    def _publish_ready(self) -> None:
        """
        active 플래그를 해제하고 로봇을 정지시킨 뒤 /loc_ready True 신호를 patrol_node에 전송합니다.
        [입력]: None
        [출력]: None (cmd_vel, loc_ready 토픽 퍼블리시)
        """
        self.active = False
        self.samples = []
        self.cmd_vel_pub.publish(Twist())  # 로봇 정지
        ready_msg = Bool()
        ready_msg.data = True
        self.loc_ready_pub.publish(ready_msg)
        self.get_logger().info('/loc_ready True 발행')

    def _average_transforms(self, transforms: List[np.ndarray]) -> np.ndarray:
        """
        T_map_to_robot 4x4 변환 행렬 목록의 위치 평균과 쿼터니언 평균을 결합해 단일 행렬을 반환합니다.
        [입력]: transforms (List[np.ndarray], 4x4 동차 변환 행렬 목록)
        [출력]: np.ndarray (4x4 평균 변환 행렬)
        """
        # 위치 평균
        avg_pos = np.mean([T[:3, 3] for T in transforms], axis=0)

        # 쿼터니언 평균 (부호 통일 후 정규화)
        quats = np.array([quaternion_from_matrix(T) for T in transforms])
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        avg_q = quats.mean(axis=0)
        avg_q /= np.linalg.norm(avg_q)

        T_avg = quaternion_matrix(avg_q)  # 4x4 회전 행렬
        T_avg[:3, 3] = avg_pos
        return T_avg

    def correct_dist(self, measured_dist: float) -> float:
        """
        AprilTag 검출기가 산출한 측정 거리에 dist_scale을 곱해 실측 보정 거리로 변환합니다.
        [입력]: measured_dist (float, 검출기 원본 거리값, 미터)
        [출력]: float (보정된 거리값, 미터)
        """
        return measured_dist * self.dist_scale

    def compute_pose(self, tag_id: int, pose_R: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
        """
        검출된 태그의 회전·이동 행렬과 사전 등록된 map 기준 태그 위치를 결합해 로봇의 전역 pose를 계산합니다.
        [입력]: tag_id (int, 태그 ID), pose_R (np.ndarray, 3x3 회전 행렬), pose_t (np.ndarray, 3x1 이동 벡터)
        [출력]: np.ndarray (4x4 T_map_to_robot 동차 변환 행렬)
        """
        map_T_tag = self.tag_poses.get(tag_id, self.default_map_T_tag)

        T_tag_to_camera = np.eye(4)
        T_tag_to_camera[:3, :3] = pose_R
        T_tag_to_camera[:3, 3] = pose_t.flatten()

        T_map_to_camera = map_T_tag @ T_tag_to_camera
        T_map_to_robot = T_map_to_camera @ self.T_camera_to_robot
        return T_map_to_robot

    def to_msg(self, T: np.ndarray) -> PoseWithCovarianceStamped:
        """
        4x4 변환 행렬을 ROS2의 PoseWithCovarianceStamped 메시지로 변환하여 공분산까지 채워줍니다.
        [입력]: T (np.ndarray, 4x4 동차 변환 행렬)
        [출력]: PoseWithCovarianceStamped (AMCL 초기 위치 메시지)
        """
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = float(T[0, 3])
        msg.pose.pose.position.y = float(T[1, 3])
        msg.pose.pose.position.z = float(T[2, 3])

        q = quaternion_from_matrix(T)
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        # 공분산 (설정 필요)
        msg.pose.covariance = [
            0.05, 0.0, 0.0,   0.0,   0.0,   0.0,
            0.0,  0.05, 0.0,  0.0,   0.0,   0.0,
            0.0,  0.0,  999.0, 0.0,  0.0,   0.0,
            0.0,  0.0,  0.0,  999.0, 0.0,   0.0,
            0.0,  0.0,  0.0,  0.0,   999.0, 0.0,
            0.0,  0.0,  0.0,  0.0,   0.0,   0.1,
        ]
        return msg


def main(args: Any = None) -> None:
    """
    스크립트 실행 시 ROS2 환경을 구성하고 LocalizationNode를 가동합니다.
    [입력]: args (Any, 명령줄 인수)
    [출력]: None
    """
    rclpy.init(args=args)
    try:
        node = LocalizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
