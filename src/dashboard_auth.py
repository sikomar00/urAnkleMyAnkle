"""역할 기반 계정 로그인, 회원가입, 10분 세션과 Dash 접근 차단을 제공한다."""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from flask import abort, redirect, render_template_string, request, session, url_for

from .audit_service import (
    authenticate_account,
    log_failure,
    record_action,
    record_login_event,
    register_account,
    update_account_session,
)


SESSION_TIMEOUT = timedelta(minutes=10)


def _now_utc() -> datetime:
    """서버가 판단하는 로그인 세션의 기준 시각을 반환한다."""
    return datetime.now(timezone.utc)


def _naive_utc(value: datetime) -> datetime:
    """MySQL DATETIME에 저장할 시간대 없는 UTC 값을 만든다."""
    return value.astimezone(timezone.utc).replace(tzinfo=None)


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
    return {"expires_at_ms": int(expiry.timestamp() * 1000), "timeout_seconds": int(SESSION_TIMEOUT.total_seconds())}


def start_admin_session(login_id: str, role: str) -> dict:
    """로그인 성공 시 계정 역할을 포함한 정확히 10분짜리 세션을 시작한다."""
    expiry = _now_utc() + SESSION_TIMEOUT
    session.clear()
    session.permanent = True
    session["admin_id"] = login_id
    session["role"] = role
    session["expires_at"] = expiry.isoformat()
    update_account_session(login_id, role, _naive_utc(expiry), online=True)
    return current_session_state() or {}


def extend_admin_session() -> dict | None:
    """현재 인증된 계정의 세션을 누적하지 않고 새 10분으로 초기화한다."""
    login_id = session.get("admin_id")
    role = session.get("role", "UNKNOWN")
    if not login_id or _read_expiry() is None:
        return None
    expiry = _now_utc() + SESSION_TIMEOUT
    session["expires_at"] = expiry.isoformat()
    update_account_session(str(login_id), str(role), _naive_utc(expiry), online=True)
    record_login_event("ACT_SESSION_EXTEND", actor_id=str(login_id), actor_role=str(role),
                       source="dashboard_auth.extend_admin_session")
    return current_session_state()


def end_admin_session(reason: str, *, event_code: str = "ACT_LOGOUT", status: str = "SUCCESS") -> None:
    """로그아웃·만료 시 DB 접속 상태를 끄고 인증 로그를 남긴 뒤 서버 세션을 지운다."""
    login_id = session.get("admin_id")
    role = session.get("role", "UNKNOWN")
    if login_id:
        update_account_session(str(login_id), str(role), None, online=False)
        record_login_event(event_code, actor_id=str(login_id), actor_role=str(role), status=status,
                           block_reason=reason, source="dashboard_auth.end_admin_session")
    session.clear()


LOGIN_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>설비 모니터링 · 계정 로그인</title>
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
  .register-link { display:block; margin:16px auto 0; padding:0; width:auto; border:0; color:#425f7a;
    background:transparent; font-size:13px; text-decoration:underline; }
  .modal-backdrop { position:fixed; inset:0; display:grid; place-items:center; background:#0006; z-index:10; }
  .modal { width:min(420px,calc(100vw - 32px)); padding:24px; border-radius:12px; background:#fff;
    box-shadow:0 12px 40px #0005; text-align:center; }
  .modal p { margin:0 0 18px; color:#26323e; font-size:15px; }
  .modal button { margin:0; }
  .account-type { display:flex; gap:10px; margin:0 0 18px; }
  .account-type button { flex:1; padding:11px 8px; border:1px solid #b8c0ca; background:#fff; color:#1f2937; }
  .account-type button.selected { border-color:#2563eb; background:#2563eb; color:#fff; }
  .register-modal { text-align:left; max-height:calc(100vh - 32px); overflow:auto; }
  .register-modal h2 { margin:0 0 18px; font-size:21px; text-align:left; }
  .register-modal .message { min-height:20px; color:#b42318; text-align:center; margin:0 0 8px; }
  .register-modal .submit { background:#2563eb; margin-top:18px; }
  .register-modal .cancel { margin-top:10px; background:#fff; border:1px solid #b8c0ca; color:#1f2937; }
  small { display:block; margin-top:20px; color:#768391; text-align:center; }
</style></head><body><main>
  <div class="eyebrow">LS JUMP-UP · 설비 모니터링</div>
  <h1>계정 로그인</h1><p>등록된 계정으로 대시보드에 접속합니다.</p>
  <form method="post" action="{{ url_for('dashboard_login') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <label for="login_id">아이디</label><input id="login_id" name="login_id" maxlength="80" autocomplete="username" required autofocus>
    <label for="password">비밀번호</label><input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">로그인</button>
  </form>
  <button type="button" class="register-link" onclick="openRegister()">계정 생성</button>
</main>
{% if error %}<div class="modal-backdrop" id="login-error-modal" role="dialog" aria-modal="true"
 onclick="if(event.target===this){closeModal('login-error-modal')}"><div class="modal">
  <p>{{ error }}</p><button type="button" onclick="closeModal('login-error-modal')">확인</button>
</div></div>{% endif %}
<div class="modal-backdrop" id="register-modal" role="dialog" aria-modal="true" style="display:{% if register_open %}grid{% else %}none{% endif %};"
 onclick="if(event.target===this){closeRegister()}"><div class="modal register-modal">
  <h2>계정 생성</h2>
  <div class="account-type"><button type="button" id="user-type" class="selected" onclick="chooseRole('USER')">일반 계정 생성</button>
    <button type="button" id="admin-type" onclick="chooseRole('ADMIN')">관리자 계정 생성</button></div>
  <form method="post" action="{{ url_for('dashboard_register') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}"><input type="hidden" id="role" name="role" value="USER">
    <label for="register_name">이름</label><input id="register_name" name="name" maxlength="120" required>
    <label for="register_email">이메일</label><input id="register_email" name="email" type="email" maxlength="160" required>
    <label for="register_phone">전화번호</label><input id="register_phone" name="phone" maxlength="40" required>
    <label for="register_id">아이디</label><input id="register_id" name="login_id" maxlength="80" autocomplete="username" required>
    <label for="register_password">비밀번호</label><input id="register_password" name="password" type="password" autocomplete="new-password" required>
    <label for="register_confirm">비밀번호 확인</label><input id="register_confirm" name="password_confirm" type="password" autocomplete="new-password" required>
    <div id="invite-wrap" style="display:none"><label for="invite_code">관리인 코드</label><input id="invite_code" name="invite_code" type="password" autocomplete="off"></div>
    <p class="message">{{ register_message or '' }}</p>
    <button class="submit" type="submit">계정 생성</button>
    <button class="cancel" type="button" onclick="closeRegister()">닫기</button>
  </form>
</div></div>
{% if register_success %}<div class="modal-backdrop" id="register-success-modal" role="dialog" aria-modal="true"><div class="modal">
  <p>계정이 생성되었습니다. 로그인해 주세요.</p><button type="button" onclick="closeModal('register-success-modal')">확인</button>
</div></div>{% endif %}
<script>
function closeModal(id){const modal=document.getElementById(id);if(modal){modal.remove();}}
function openRegister(){document.getElementById('register-modal').style.display='grid';}
function closeRegister(){document.getElementById('register-modal').style.display='none';}
function chooseRole(role){document.getElementById('role').value=role;const admin=role==='ADMIN';
 document.getElementById('user-type').classList.toggle('selected',!admin);document.getElementById('admin-type').classList.toggle('selected',admin);
 document.getElementById('invite-wrap').style.display=admin?'block':'none';document.getElementById('invite_code').required=admin;}
{% if register_role == 'ADMIN' %}chooseRole('ADMIN');{% endif %}
</script></body></html>"""


def install_auth(server) -> None:
    """인증되지 않은 요청이 Dash 데이터 콜백에 접근하지 못하게 한다."""
    server.secret_key = os.environ.get("DASHBOARD_SESSION_SECRET") or secrets.token_urlsafe(48)
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("DASHBOARD_COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=SESSION_TIMEOUT,
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
            end_admin_session("10분 세션 만료", event_code="ACT_SESSION_EXPIRED", status="TIMEOUT")
            if request.path.startswith(("/_dash", "/assets/")):
                abort(401)
            return redirect(url_for("dashboard_login"))
        if request.path == "/login" or request.path == "/register" or admin_id:
            return None
        if request.path.startswith(("/_dash", "/assets/")):
            abort(401)
        return redirect(url_for("dashboard_login"))

    def render_login(*, error: str | None = None, register_open: bool = False,
                     register_message: str | None = None, register_success: bool = False,
                     register_role: str = "USER", status: int = 200):
        if "login_csrf" not in session:
            session["login_csrf"] = secrets.token_urlsafe(32)
        return render_template_string(
            LOGIN_HTML, csrf_token=session["login_csrf"], error=error,
            register_open=register_open, register_message=register_message,
            register_success=register_success, register_role=register_role,
        ), status

    @server.route("/login", methods=["GET", "POST"])
    def dashboard_login():
        if session.get("admin_id"):
            return redirect("/")
        if request.method == "GET":
            return render_login()
        sent = request.form.get("csrf_token", "")
        if not hmac.compare_digest(session.get("login_csrf", ""), sent):
            abort(400)
        login_id = request.form.get("login_id", "").strip()[:80]
        password = request.form.get("password", "")
        try:
            role = authenticate_account(login_id, password)
        except Exception as exc:
            log_failure("ACT_LOGIN", "계정 로그인", exc, actor_id=None, source="dashboard_auth.dashboard_login")
            return render_login(error="로그인 확인 중 오류가 발생했습니다.", status=503)
        if role:
            start_admin_session(login_id, role)
            record_login_event("ACT_LOGIN", actor_id=login_id, actor_role=role, source="dashboard_auth.dashboard_login")
            return redirect("/")
        record_login_event("ACT_LOGIN_BLOCKED", actor_id=login_id or "ANONYMOUS", actor_role="UNKNOWN",
                           status="BLOCKED", block_reason="아이디 또는 비밀번호 불일치",
                           source="dashboard_auth.dashboard_login")
        return render_login(error="아이디 또는 비밀번호가 올바르지 않습니다.", status=401)

    @server.route("/register", methods=["POST"])
    def dashboard_register():
        sent = request.form.get("csrf_token", "")
        if not hmac.compare_digest(session.get("login_csrf", ""), sent):
            abort(400)
        role = request.form.get("role", "USER").upper()
        login_id = request.form.get("login_id", "").strip()[:80]
        try:
            register_account(
                role=role, login_id=login_id, password=request.form.get("password", ""),
                password_confirm=request.form.get("password_confirm", ""), name=request.form.get("name", ""),
                phone=request.form.get("phone", ""), email=request.form.get("email", ""),
                invite_code=request.form.get("invite_code", ""),
            )
        except (ValueError, PermissionError, FileExistsError) as exc:
            # 오류 안내에는 비밀값을 되돌려 보여 주지 않는다.
            record_action("ACT_ACCOUNT_REGISTER_BLOCKED", "계정 생성 차단", actor_id=login_id,
                          actor_role="UNKNOWN", target_type="account", target_id=login_id,
                          status="BLOCKED", block_reason=type(exc).__name__)
            return render_login(register_open=True, register_message=str(exc), register_role=role, status=400)
        except Exception as exc:
            log_failure("ACT_ACCOUNT_REGISTER", "계정 생성", exc, actor_id=login_id,
                        source="dashboard_auth.dashboard_register")
            return render_login(register_open=True, register_message="계정 생성 중 오류가 발생했습니다.",
                                register_role=role, status=503)
        return render_login(register_success=True)
