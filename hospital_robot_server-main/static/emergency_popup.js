/**
 * emergency_popup.js
 * 모든 페이지에서 공통으로 동작하는 응급 알림 팝업
 *
 * 동작 방식:
 *   - /api/msg_test 를 2초마다 폴링
 *   - data.emergency === "위급상황" 이면 팝업 표시
 *   - [코드 블루 출동] → /api/robot/code_blue POST → robot6을 robot1 위치로 출동
 *   - [확인] 버튼 → 팝업 닫기 (상황 지속 시 다음 폴링에서 재표시)
 */

(function () {
    'use strict';

    /* ── 설정 ── */
    const EMERGENCY_TRIGGER = '위급상황';
    const POLL_INTERVAL_MS  = 2000;

    /* ── 상태 ── */
    let _popupActive      = false;
    let _alertShown       = false;
    let _pollTimer        = null;
    let _codeBlueLoading  = false;   // 코드 블루 중복 호출 방지

    /* ── 스타일 + 팝업 HTML 1회 삽입 ── */
    function _injectPopup() {
        if (document.getElementById('_emg-overlay')) return;

        /* 애니메이션 스타일 */
        if (!document.getElementById('_emg-style')) {
            const style = document.createElement('style');
            style.id = '_emg-style';
            style.textContent = `
                @keyframes _emg-shake {
                    0%,100%{ transform:translateX(0); }
                    20%    { transform:translateX(-8px); }
                    40%    { transform:translateX(8px); }
                    60%    { transform:translateX(-6px); }
                    80%    { transform:translateX(6px); }
                }
                #_emg-overlay * { box-sizing: border-box; }
                #_emg-code-blue-btn:disabled {
                    opacity: 0.6; cursor: not-allowed;
                }
                #_emg-result {
                    margin-top: 12px;
                    font-size: 0.88rem;
                    min-height: 20px;
                    border-radius: 6px;
                    padding: 6px 10px;
                }
                #_emg-result.success { background:#e8f5e9; color:#2e7d32; }
                #_emg-result.error   { background:#fce4ec; color:#c62828; }
            `;
            document.head.appendChild(style);
        }

        /* 팝업 오버레이 */
        const overlay = document.createElement('div');
        overlay.id = '_emg-overlay';
        overlay.style.cssText = [
            'display:none',
            'position:fixed',
            'inset:0',
            'background:rgba(0,0,0,0.78)',
            'z-index:99999',
            'justify-content:center',
            'align-items:center',
        ].join(';');

        overlay.innerHTML = `
            <div id="_emg-box" style="
                background:#fff;
                width:440px;
                border-radius:16px;
                border:5px solid #d32f2f;
                padding:36px 32px 28px;
                text-align:center;
                box-shadow:0 8px 32px rgba(0,0,0,0.4);
                animation:_emg-shake 0.4s ease;
            ">
                <div style="font-size:3.2rem; margin-bottom:10px;">🚨</div>
                <h2 style="color:#d32f2f; margin:0 0 8px; font-size:1.5rem; font-weight:800;">
                    응급 상황 발생!
                </h2>
                <p style="color:#555; margin:0 0 6px; font-size:0.97rem; line-height:1.6;">
                    환자에게 응급 상황이 발생했습니다.<br>
                    의료진은 즉시 확인하십시오.
                </p>

                <!-- 환자 위치 표시 -->
                <p id="_emg-location" style="
                    color:#d32f2f; font-size:1rem; font-weight:bold;
                    background:#ffeaea; border-radius:8px; padding:8px 14px;
                    margin:10px 0 18px; min-height:38px;
                "></p>

                <!-- 버튼 영역 -->
                <div style="display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">

                    <!-- 코드 블루 출동 버튼 -->
                    <button id="_emg-code-blue-btn" style="
                        background:#1565c0; color:#fff; border:none;
                        border-radius:8px; padding:12px 24px;
                        font-size:0.97rem; font-weight:bold; cursor:pointer;
                        transition:background 0.2s;
                        box-shadow:0 3px 10px rgba(21,101,192,0.35);
                    ">🚑 코드 블루 출동</button>

                    <!-- 확인 버튼 -->
                    <button id="_emg-confirm-btn" style="
                        background:#d32f2f; color:#fff; border:none;
                        border-radius:8px; padding:12px 24px;
                        font-size:0.97rem; font-weight:bold; cursor:pointer;
                        transition:background 0.2s;
                    ">확인</button>

                </div>

                <!-- 코드 블루 결과 메시지 -->
                <div id="_emg-result"></div>
            </div>`;

        /* 코드 블루 버튼 이벤트 */
        overlay.querySelector('#_emg-code-blue-btn').addEventListener('click', _triggerCodeBlue);

        /* 확인 버튼 이벤트 */
        overlay.querySelector('#_emg-confirm-btn').addEventListener('click', _closePopup);

        document.body.appendChild(overlay);
    }

    /* ── 팝업 열기 ── */
    function _openPopup(locationText) {
        _injectPopup();
        const overlay = document.getElementById('_emg-overlay');
        const locEl   = document.getElementById('_emg-location');
        const result  = document.getElementById('_emg-result');
        const btn     = document.getElementById('_emg-code-blue-btn');

        if (locEl)    locEl.textContent   = locationText || '';
        if (result)   result.textContent  = '';
        if (result)   result.className    = '';
        if (btn)      btn.disabled        = false;
        if (btn)      btn.textContent     = '🚑 코드 블루 출동';

        overlay.style.display = 'flex';
        _popupActive     = true;
        _codeBlueLoading = false;

        /* 흔들림 재실행 */
        const box = document.getElementById('_emg-box');
        if (box) {
            box.style.animation = 'none';
            box.offsetHeight;
            box.style.animation = '_emg-shake 0.4s ease';
        }
    }

    /* ── 팝업 닫기 (서버에 confirm 전송) ── */
    function _closePopup() {
        const overlay = document.getElementById('_emg-overlay');
        if (overlay) overlay.style.display = 'none';
        _popupActive  = false;
        _alertShown   = false;

        // 서버 위급상황 상태 리셋
        fetch('/api/emergency/confirm', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    '{}'
        }).catch(() => {});
    }

    /* ── 코드 블루 출동 ── */
    function _triggerCodeBlue() {
        if (_codeBlueLoading) return;
        _codeBlueLoading = true;

        const btn    = document.getElementById('_emg-code-blue-btn');
        const result = document.getElementById('_emg-result');

        if (btn) {
            btn.disabled    = true;
            btn.textContent = '⏳ 출동 명령 전송 중...';
        }
        if (result) {
            result.textContent = '';
            result.className   = '';
        }

        fetch('/api/robot/code_blue', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    '{}'
        })
        .then(r => {
            // 401: 세션 만료 → 로그인 페이지로 이동
            if (r.status === 401) {
                if (result) {
                    result.textContent = '⚠️ 세션이 만료되었습니다. 다시 로그인하세요.';
                    result.className   = 'error';
                }
                if (btn) {
                    btn.disabled    = false;
                    btn.textContent = '🚑 코드 블루 출동';
                }
                _codeBlueLoading = false;
                setTimeout(() => { window.location.href = '/login'; }, 1500);
                return null;
            }
            return r.json();
        })
        .then(res => {
            if (res === null) return;   // 401 처리 후 종료
            if (res.success) {
                const d   = res.data;
                const src = d.source === 'amcl' ? 'AMCL' : 'Odom';
                const modeBadge = d.mode === 'mock'
                    ? ' <span style="color:#f57c00;">[Mock 모드]</span>'
                    : '';
                if (result) {
                    result.innerHTML = `✅ robot6 출동 명령 전송 완료<br>robot1 위치: (${d.x}, ${d.y}) [${src}]${modeBadge}`;
                    result.className = 'success';
                }
                if (btn) btn.textContent = d.mode === 'mock' ? '✅ 전송됨 (Mock)' : '✅ 출동 완료';
            } else {
                throw new Error(res.error || '출동 실패');
            }
        })
        .catch(err => {
            if (result) {
                result.textContent = `❌ ${err.message}`;
                result.className   = 'error';
            }
            if (btn) {
                btn.disabled    = false;
                btn.textContent = '🚑 코드 블루 출동';
            }
            _codeBlueLoading = false;
        });
    }

    /* ── /api/msg_test 폴링 ── */
    function _poll() {
        fetch('/api/msg_test')
            .then(r => r.json())
            .then(result => {
                if (!result.success) return;

                const emergency = result.data.emergency;
                const location  = result.data.location || '';

                if (emergency === EMERGENCY_TRIGGER) {
                    // 팝업이 열려 있으면 위치 텍스트만 갱신
                    if (_popupActive) {
                        const locEl = document.getElementById('_emg-location');
                        if (locEl && location) locEl.textContent = location;
                    } else if (!_alertShown) {
                        _alertShown = true;
                        _openPopup(location);
                    }
                } else {
                    // 서버가 '대기 중'이면 팝업도 닫기
                    if (_popupActive) _closePopup();
                    _alertShown = false;
                }
            })
            .catch(() => {});
    }

    /* ── 폴링 시작 / 탭 비활성 시 중단 ── */
    function _startPolling() {
        if (_pollTimer) return;
        _poll();
        _pollTimer = setInterval(_poll, POLL_INTERVAL_MS);
    }

    function _stopPolling() {
        clearInterval(_pollTimer);
        _pollTimer = null;
    }

    document.addEventListener('visibilitychange', () => {
        document.hidden ? _stopPolling() : _startPolling();
    });

    /* ── DOMContentLoaded 후 시작 ── */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _startPolling);
    } else {
        _startPolling();
    }

})();
