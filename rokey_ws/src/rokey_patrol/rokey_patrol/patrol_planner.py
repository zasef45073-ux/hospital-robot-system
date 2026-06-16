#!/usr/bin/env python3
"""
Patrol planner that cycles between 'hall' and 'room2'.

변경 알고리즘:
- hall 도착: 잠깐 대기 후 다음 목적지 이동
- room2 도착:
    1) detect_enable=True 발행
    2) amr_detect가 360도 탐색 수행
    3) search_done 토픽에서 "문제없음" 수신
    4) detect_enable=False 발행
    5) 다음 목적지로 이동
"""

import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger

from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String


class PatrolPlanner(Node):
    def __init__(self):
        super().__init__('patrol_planner_node')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 네임스페이스 /robot1로 실행하면 자동으로 /robot1/detect_enable
        self.detect_pub = self.create_publisher(Bool, 'detect_enable', 10)

        # 네임스페이스 /robot1로 실행하면 자동으로 /robot1/search_done 구독
        self.search_done_sub = self.create_subscription(
            String,
            'search_done',
            self.search_done_callback,
            10
        )

        # Camera start service client
        # 기존 코드처럼 절대경로 유지: /robot1/oakd/start_camera
        self.camera_client = self.create_client(Trigger, '/robot1/oakd/start_camera')

        self.waypoints = {
            'room2': (-1.774,  1.585, -0.116),
            'room2_front': (-2.767,  1.448, -0.116),
            'hall':  (-2.814,  0.081,  1.900),
        }

        self.sequence = ['hall', 'room2_front', 'room2']
        self.current_idx = 0

        self.hall_dwell_sec = 2.0

        # room2에서 search_done을 기다리는 상태인지 확인하는 플래그
        self.waiting_search_done = False

        self.get_logger().info(
            'Patrol planner started. Starting camera then departing in 3s...'
        )

        self.timer = self.create_timer(3.0, self.start_patrol)

    def _start_camera(self):
        if not self.camera_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('start_camera service not available, skipping.')
            return

        future = self.camera_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None:
            self.get_logger().info(f'Camera started: {future.result().message}')
        else:
            self.get_logger().warn('Camera start call failed.')

    def start_patrol(self):
        self.timer.cancel()
        self._start_camera()
        self._set_detection(False)
        self.send_next_goal()

    def _set_detection(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.detect_pub.publish(msg)
        self.get_logger().info(f'detect_enable = {enable}')

    def send_next_goal(self):
        name = self.sequence[self.current_idx]
        x, y, yaw = self.waypoints[name]

        self.get_logger().info(
            f'Navigating to "{name}" (x={x:.3f}, y={y:.3f}, yaw={yaw:.2f})'
        )

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Nav2 action server not available. Retrying in 2s...'
            )
            self.create_timer(2.0, self._retry_once)
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._send_goal_future = self.nav_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def _retry_once(self):
        self.send_next_goal()

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal.')
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        name = self.sequence[self.current_idx]

        if status == 4:
            self.get_logger().info(f'Arrived at "{name}".')

            if name == 'room2':
                # room2에서는 시간 대기하지 않고,
                # amr_detect의 360도 탐색 완료 토픽을 기다림
                self.waiting_search_done = True
                self._set_detection(True)
                self.get_logger().info(
                    'Detection ON - waiting for search_done topic from amr_detect.'
                )
                return

            else:
                self._set_detection(False)
                self.get_logger().info(
                    f'Dwelling {self.hall_dwell_sec:.1f}s at hall.'
                )
                time.sleep(self.hall_dwell_sec)

        else:
            self.get_logger().warn(
                f'Navigation to "{name}" failed (status={status}). Moving on.'
            )

        self._move_to_next_goal()

    def search_done_callback(self, msg: String):
        if not self.waiting_search_done:
            return

        if msg.data != '문제없음':
            self.get_logger().info(f'Ignored search_done message: {msg.data}')
            return

        self.get_logger().info(
            'Search done received: 문제없음. Detection OFF and leaving room2.'
        )

        self.waiting_search_done = False
        self._set_detection(False)

        self._move_to_next_goal()

    def _move_to_next_goal(self):
        self.current_idx = (self.current_idx + 1) % len(self.sequence)
        self.send_next_goal()


def main(args=None):
    rclpy.init(args=args)
    node = PatrolPlanner()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node._set_detection(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

