"""한 명의 관리자 계정으로 Dash 접근을 통제하는 Flask 로그인 화면."""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from flask import abort, redirect, render_template_string, request, session, url_for

from .audit_service import log_failure, record_action, verify_admin


SESSION_TIMEOUT = timedelta(minutes=10)


def _now_utc() -> datetime:
    """서버가 판단하는 로그인 세션의 기준 시각을 반환한다."""
    return datetime.now(timezone.utc)


def _read_expiry() -> datetime | None:
    """서명된 Flask 세션의 만료 시각을 안전하게 읽는다."""
    raw = session.get("expires_at")
    if not isinstance(raw, str):
        return None
    try:
        expiry = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return expiry if expiry.tzinfo is not None else expiry.replace(tzinfo=timezone.utc)


def current_session_state() -> dict | None:
    """브라우저 카운트다운에 필요한 공개 가능한 세션 정보만 반환한다."""
    expiry = _read_expiry()
    if not session.get("admin_id") or expiry is None:
        return None
    return {
        "expires_at_ms": int(expiry.timestamp() * 1000),
        "timeout_seconds": int(SESSION_TIMEOUT.total_seconds()),
    }


def start_admin_session(login_id: str) -> dict:
    """로그인 성공 시 기존 세션을 비우고 정확히 10분짜리 세션을 시작한다."""
    session.clear()
    session.permanent = True
    session["admin_id"] = login_id
    session["expires_at"] = (_now_utc() + SESSION_TIMEOUT).isoformat()
    return current_session_state() or {}


def extend_admin_session() -> dict | None:
    """현재 인증된 관리자만 버튼으로 10분 세션을 다시 시작할 수 있다."""
    login_id = session.get("admin_id")
    if not login_id or _read_expiry() is None:
        return None
    session["expires_at"] = (_now_utc() + SESSION_TIMEOUT).isoformat()
    record_action("ACT_SESSION_EXTEND", "로그인 시간 연장", actor_id=str(login_id),
                  target_type="session", target_id=str(login_id))
    return current_session_state()


LOGIN_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>설비 모니터링 · 관리자 로그인</title>
<style>
  * { box-sizing:border-box; } body { margin:0; min-height:100vh; display:grid; place-items:center;
    background:#f3f5f7; color:#19232e; font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif; }
  main { width:min(400px,calc(100vw - 32px)); padding:34px; background:#fff; border:1px solid #d7dde3;
    border-radius:16px; box-shadow:0 16px 40px #1b283b14; }
  .eyebrow { color:#637181; font-size:12px; font-weight:700; letter-spacing:.08em; }
  h1 { margin:8px 0 6px; font-size:25px; } p { margin:0 0 25px; color:#627080; font-size:13px; }
  label { display:block; margin:16px 0 6px; font-size:13px; font-weight:700; }
  input { width:100%; padding:12px; border:1px solid #bac5cf; border-radius:8px; font:inherit; }
  input:focus { outline:2px solid #235a96; outline-offset:1px; }
  button { width:100%; margin-top:25px; padding:12px; border:0; border-radius:8px;
    color:#fff; background:#20364c; font:inherit; font-weight:700; cursor:pointer; }
  .modal-backdrop { position:fixed; inset:0; display:grid; place-items:center; background:#0006; z-index:10; }
  .modal { width:min(340px,calc(100vw - 32px)); padding:24px; border-radius:12px; background:#fff;
    box-shadow:0 12px 40px #0005; text-align:center; }
  .modal p { margin:0 0 18px; color:#26323e; font-size:15px; }
  .modal button { margin:0; }
  small { display:block; margin-top:20px; color:#768391; text-align:center; }
</style></head><body><main>
  <div class="eyebrow">LS JUMP-UP · 설비 모니터링</div>
  <h1>관리자 로그인</h1><p>등록된 관리자 계정으로 대시보드에 접속합니다.</p>
  <form method="post" action="{{ url_for('dashboard_login') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <label for="login_id">아이디</label><input id="login_id" name="login_id" maxlength="80" autocomplete="username" required autofocus>
    <label for="password">비밀번호</label><input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">로그인</button>
  </form>
  <small>관리자 1명 전용 · 회원가입 없음</small>
</main>
{% if error %}<div class="modal-backdrop" id="login-error-modal" role="dialog" aria-modal="true"
 onclick="if(event.target===this){closeLoginError()}"><div class="modal">
  <p>{{ error }}</p><button type="button" onclick="closeLoginError()">확인</button>
</div></div>{% endif %}
<script>function closeLoginError(){const modal=document.getElementById('login-error-modal');if(modal){modal.remove();}}</script>
</body></html>"""


def install_auth(server) -> None:
    """인증되지 않은 요청이 Dash 데이터 콜백에 접근하지 못하게 한다."""
    # 로컬 개발에서 .env가 빠져도 예측하기 어려운 임시 키를 사용한다.
    # 운영·재시작 간 세션 유지에는 .env의 고정된 무작위 키가 필요하다.
    server.secret_key = os.environ.get("DASHBOARD_SESSION_SECRET") or secrets.token_urlsafe(48)
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("DASHBOARD_COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=SESSION_TIMEOUT,
        # 일반 Dash 요청이 올 때마다 쿠키 만료가 자동으로 늘어나지 않게 한다.
        SESSION_REFRESH_EACH_REQUEST=False,
    )

    @server.after_request
    def prevent_private_page_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @server.before_request
    def require_admin_session():
        admin_id = session.get("admin_id")
        expiry = _read_expiry()
        if admin_id and (expiry is None or expiry <= _now_utc()):
            # 세션을 지우므로 이후 요청에서도 같은 만료 로그가 중복 저장되지 않는다.
            session.clear()
            record_action("ACT_SESSION_EXPIRED", "로그인 세션 만료", actor_id=str(admin_id),
                          target_type="session", target_id=str(admin_id),
                          status="TIMEOUT", block_reason="10분 세션 만료")
            if request.path.startswith(("/_dash", "/assets/")):
                abort(401)
            return redirect(url_for("dashboard_login"))
        if request.path == "/login" or admin_id:
            return None
        if request.path.startswith(("/_dash", "/assets/")):
            abort(401)
        return redirect(url_for("dashboard_login"))

    @server.route("/login", methods=["GET", "POST"])
    def dashboard_login():
        if session.get("admin_id"):
            return redirect("/")
        if "login_csrf" not in session:
            session["login_csrf"] = secrets.token_urlsafe(32)
        error = None
        status = 200
        if request.method == "POST":
            sent = request.form.get("csrf_token", "")
            if not hmac.compare_digest(session["login_csrf"], sent):
                abort(400)
            login_id = request.form.get("login_id", "").strip()[:80]
            password = request.form.get("password", "")
            try:
                authenticated = bool(login_id and password and verify_admin(login_id, password))
            except Exception as exc:
                log_failure("ACT_LOGIN", "관리자 로그인", exc, actor_id=None,
                            source="dashboard_auth.dashboard_login")
                error, status = "로그인 확인 중 오류가 발생했습니다.", 503
            else:
                if authenticated:
                    start_admin_session(login_id)
                    record_action("ACT_LOGIN", "관리자 로그인", target_type="admin",
                                  target_id=login_id)
                    return redirect("/")
                record_action("ACT_LOGIN", "관리자 로그인", status="BLOCKED",
                              block_reason="아이디 또는 비밀번호 불일치")
                error, status = "아이디 또는 비밀번호가 올바르지 않습니다.", 401
        return render_template_string(LOGIN_HTML, csrf_token=session["login_csrf"], error=error), status
