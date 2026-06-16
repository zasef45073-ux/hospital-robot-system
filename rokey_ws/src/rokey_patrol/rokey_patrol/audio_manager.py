#!/usr/bin/env python3
"""
================================================================================
  음성(mp3) 재생 관리자 모듈
================================================================================

[전체 개요]
이 모듈은 ROS 노드(amr_detect.py)에서 상황별로 mp3 파일을 틀어주기 위한 도구입니다.
- "일어나세요"(sit), "감사합니다"(thanks), "이상 없음"(clear), "도와주세요"(help)
  등 4가지 음성을 이름으로 골라서 재생할 수 있게 해줍니다.
- 시스템에 깔린 외부 플레이어(ffplay/mpv/cvlc)를 subprocess로 실행해 음성을 재생합니다.
  (즉, 파이썬 안에서 직접 디코딩하지 않고 OS의 명령행 플레이어를 빌려 씀)
- 비동기 재생(play_once_async)이 기본이고, 필요 시 동기 재생(play_once_blocking)도 가능.

[클래스 구조]
- AudioPlayer  : mp3 파일 1개에 대응. 시작/정지/재생 담당
- AudioManager : 여러 AudioPlayer들을 이름('sit'/'help'/'thanks'/'clear')으로 묶어서 관리
                 'sit'에는 추가로 "일정 시간 뒤 자동 정지" 타이머를 붙임
                 → 누운 사람에게 안내 음성을 너무 길게 틀지 않기 위함

[왜 외부 플레이어를 쓰나?]
파이썬에서 직접 mp3를 재생하려면 라이브러리 의존(예: pygame, simpleaudio)이 생기지만,
리눅스에는 ffplay/mpv/cvlc 중 하나는 거의 깔려 있어서 가볍게 처리할 수 있습니다.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 표준 라이브러리만 사용 (외부 의존성 없음)
# ─────────────────────────────────────────────────────────────────────────────
import shutil      # shutil.which : 시스템 PATH에서 실행 파일을 찾는 용도
import subprocess  # 외부 명령(플레이어)을 자식 프로세스로 띄우기 위함
import threading   # 비동기 재생 / 자동 정지 타이머에 사용


# ─────────────────────────────────────────────────────────────────────────────
# 사용 가능한 외부 플레이어 후보 목록
#   각 항목 = (실행 파일 이름, 그 플레이어를 "조용히 한 번만" 재생시키는 옵션들)
#
#   - ffplay : ffmpeg에 딸려오는 가장 흔한 도구. -nodisp(영상창 숨김), -autoexit(끝나면 종료)
#   - mpv    : 유명한 미디어 플레이어. --no-video(영상 끔), --really-quiet(로그 숨김)
#   - cvlc   : VLC의 콘솔 버전. --play-and-exit(끝나면 종료), --quiet(조용히)
#
#   리스트의 "앞쪽이 우선순위"입니다. 시스템에 ffplay가 있으면 그걸 쓰고,
#   없으면 mpv, 그것도 없으면 cvlc 순서로 선택됩니다.
# ─────────────────────────────────────────────────────────────────────────────
_PLAYER_CANDIDATES = [
    ('ffplay', ['-nodisp', '-autoexit', '-loglevel', 'quiet']),
    ('mpv',    ['--no-video', '--really-quiet']),
    ('cvlc',   ['--play-and-exit', '--quiet']),
]


# ─────────────────────────────────────────────────────────────────────────────
# AudioPlayer
#   mp3 파일 1개에 대응하는 "한 줄짜리" 플레이어.
#   - 만들 때 어떤 외부 플레이어를 쓸지 자동으로 결정 (_find_player)
#   - 동기/비동기 재생, 정지 기능 제공
#   - 멀티스레드에서 동시에 stop/play가 호출돼도 안전하도록 락(_lock) 사용
# ─────────────────────────────────────────────────────────────────────────────
class AudioPlayer:
    def __init__(self, audio_path: str, logger=None):
        self.audio_path = audio_path
        self.logger = logger

        # 이 파일을 재생할 때 실제로 호출할 명령(리스트). 예) ['ffplay', '-nodisp', ..., '/path/x.mp3']
        # 시스템에 적합한 플레이어가 하나도 없으면 None
        self.cmd = self._find_player(audio_path)

        self._thread = None  # 비동기 재생 중인 스레드 핸들
        self._proc   = None  # 현재 돌아가는 자식 프로세스(subprocess.Popen 결과)
        self._lock   = threading.Lock()  # _proc/_thread 동시 접근 방지

    # ─────────────────────────────────────────────────────────────────
    # 사용 가능한 첫 번째 플레이어를 찾아 명령 리스트를 만들어 반환.
    # 못 찾으면 None.
    #
    # shutil.which(exe) 는 PATH에서 실행 파일을 찾으면 경로를, 없으면 None을 줌.
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _find_player(path: str):
        for exe, args in _PLAYER_CANDIDATES:
            if shutil.which(exe):
                # 예: ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', '/path/x.mp3']
                return [exe, *args, path]
        return None

    # logger가 주입돼 있을 때만 경고 출력 (없을 때 그냥 무시)
    def _warn(self, msg: str):
        if self.logger:
            self.logger.warn(msg)

    # ─────────────────────────────────────────────────────────────────
    # 현재 돌고 있는 자식 프로세스를 깔끔하게 종료시킴.
    #   1) terminate() - SIGTERM 보내고 1초 기다림
    #   2) 그래도 안 죽으면 kill() - SIGKILL로 강제 종료
    #   3) 어떤 경우든 _proc 은 None으로 정리
    # ─────────────────────────────────────────────────────────────────
    def _terminate_proc(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=1.0)
        except Exception:
            # terminate가 실패하거나 1초 안에 안 죽었으면 강제 kill
            try:
                self._proc.kill()
            except Exception:
                pass  # 그래도 안 되면 그냥 포기 (로그도 남기지 않음)
        finally:
            self._proc = None

    # 외부에서 호출하는 정지 메서드. 락을 잡고 안전하게 종료
    def stop(self):
        with self._lock:
            self._terminate_proc()

    # ─────────────────────────────────────────────────────────────────
    # 동기(blocking) 재생: 이 함수 호출은 음성이 끝날 때까지 리턴하지 않음.
    #   - "이상 없음" 음성처럼 끝까지 들려준 뒤에 다음 동작을 하고 싶을 때 사용.
    # ─────────────────────────────────────────────────────────────────
    def play_once_blocking(self):
        if self.cmd is None:
            self._warn(f'No player for {self.audio_path}')
            return
        try:
            # stdout/stderr를 /dev/null로 버려 콘솔이 지저분해지지 않게 함
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._proc.wait()  # 여기서 음성이 끝날 때까지 멈춰있음
        except Exception as e:
            self._warn(f'Play failed ({self.audio_path}): {e}')
        finally:
            self._proc = None

    # ─────────────────────────────────────────────────────────────────
    # 비동기 재생: 별도의 스레드에서 play_once_blocking 을 돌림.
    #   - 호출 즉시 리턴하고 음성은 백그라운드에서 흘러감.
    #   - 이미 재생 중이면 새로 시작하지 않음 (중복 재생 방지)
    #   - daemon=True 라서 메인 프로그램이 종료되면 같이 사라짐
    # ─────────────────────────────────────────────────────────────────
    def play_once_async(self):
        if self.cmd is None:
            self._warn(f'No player for {self.audio_path}')
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                # 아직 이전 재생이 진행 중이면 그대로 둠
                return
            self._thread = threading.Thread(target=self.play_once_blocking, daemon=True)
            self._thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# AudioManager
#   여러 mp3 파일을 이름으로 묶어 관리.
#     'sit'    : 일어나달라고 안내하는 음성
#     'help'   : 위급상황 도움 요청 음성
#     'thanks' : 회복했을 때 감사 음성
#     'clear'  : 360도 스캔 후 "이상 없음" 음성
#
#   특별 동작 - 'sit' 음성은:
#     - 새로 play('sit') 할 때마다 이전 sit 재생을 정지하고 다시 시작
#     - ask_sit_timeout_sec(기본 10초) 후에 자동으로 멈추도록 타이머 설정
#       → "일어나세요"가 무한히 반복되지 않게 보장
# ─────────────────────────────────────────────────────────────────────────────
class AudioManager:
    def __init__(
        self,
        sit_audio_path: str,
        thanks_audio_path: str,
        clear_audio_path: str,
        help_audio_path: str,
        ask_sit_timeout_sec: float = 10.0,
        logger=None,
    ):
        self.logger = logger
        self.ask_sit_timeout_sec = ask_sit_timeout_sec

        # 이름 → AudioPlayer 매핑 테이블
        # play('sit') 처럼 이름만 알면 어떤 파일이든 동일한 인터페이스로 다룰 수 있음
        self._players = {
            'sit':    AudioPlayer(sit_audio_path,    logger),
            'help':   AudioPlayer(help_audio_path,   logger),
            'thanks': AudioPlayer(thanks_audio_path, logger),
            'clear':  AudioPlayer(clear_audio_path,  logger),
        }

        # 'sit' 음성을 일정 시간 후 자동으로 멈추기 위한 타이머 핸들
        # (현재 활성화된 타이머가 있을 때만 not None)
        self._sit_stop_timer = None

    # ─────────────────────────────────────────────────────────────────
    # sit 자동 정지 타이머가 살아있다면 취소.
    # 타이머 객체는 cancel() 후 한 번 쓰고 버림(재사용하지 않음).
    # ─────────────────────────────────────────────────────────────────
    def _cancel_sit_timer(self):
        if self._sit_stop_timer is not None:
            self._sit_stop_timer.cancel()
            self._sit_stop_timer = None

    # ─────────────────────────────────────────────────────────────────
    # 이름으로 음성 재생.
    #   block=True : 끝까지 듣고 리턴 (현재 'clear' 음성에서 사용)
    #   block=False(기본) : 백그라운드 재생 후 즉시 리턴
    #
    # 'sit' 만 특별 처리:
    #   - 이전 sit 재생을 멈추고
    #   - 새로 비동기 재생을 시작하고
    #   - N초 뒤 자동 정지 타이머를 건다
    # ─────────────────────────────────────────────────────────────────
    def play(self, name: str, block: bool = False):
        player = self._players[name]

        if name == 'sit':
            # 이전에 걸린 자동 정지 타이머가 있으면 우선 취소 (덮어쓰기 위해)
            self._cancel_sit_timer()
            # 진행 중이던 sit 재생도 일단 멈추고 새로 시작
            player.stop()
            player.play_once_async()

            # ask_sit_timeout_sec 후에 player.stop() 을 호출하는 1회용 타이머
            # daemon=True 이므로 메인 프로그램 종료 시 같이 정리됨
            self._sit_stop_timer = threading.Timer(self.ask_sit_timeout_sec, player.stop)
            self._sit_stop_timer.daemon = True
            self._sit_stop_timer.start()
            return

        # 그 외 음성들은 단순 동기/비동기 재생만
        if block:
            player.play_once_blocking()
        else:
            player.play_once_async()

    # ─────────────────────────────────────────────────────────────────
    # 특정 이름 음성만 정지. 'sit' 이면 자동 정지 타이머도 같이 취소.
    # ─────────────────────────────────────────────────────────────────
    def stop(self, name: str):
        if name == 'sit':
            self._cancel_sit_timer()
        self._players[name].stop()

    # 모든 음성을 일괄 정지 (노드가 꺼지거나 비활성화될 때 사용)
    def stop_all(self):
        self._cancel_sit_timer()
        for p in self._players.values():
            p.stop()
