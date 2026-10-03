"""역할 기반 계정 로그인, 회원가입, 10분 세션과 Dash 접근 차단을 제공한다."""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import abort, redirect, render_template_string, request, session, url_for

from .audit_service import (
    account_identity_for_session,
    authenticate_account,
    log_failure,
    record_action,
    record_login_event,
    register_account,
    update_account_session,
)


SESSION_TIMEOUT = timedelta(minutes=10)

# 로그인 전에는 /assets 요청이 401이므로(require_admin_session) 대시보드와 같은
# 토큰·컴포넌트 CSS를 로그인 페이지에 직접 넣는다. 색은 이 파일에 적지 않는다.
_ASSETS = Path(__file__).resolve().parents[1] / "assets"
DESIGN_CSS = "\n".join((_ASSETS / name).read_text(encoding="utf-8")
                        for name in ("00-tokens.css", "01-type.css", "02-bundle.css", "03-app.css"))


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
    identity = account_identity_for_session(login_id)
    if identity is None or identity[1] != role:
        raise PermissionError("로그인 계정을 확인할 수 없습니다.")
    expiry = _now_utc() + SESSION_TIMEOUT
    session.clear()
    session.permanent = True
    session["admin_id"] = login_id
    session["role"] = role
    session["account_pk"] = identity[0]
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
<html lang="ko" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>설비 모니터링 · 계정 로그인</title>
<style>{{ design_css|safe }}</style>
<style>
  /* 로그인 화면 배치만 적는다 — 색·글꼴·테두리는 위 토큰 CSS의 클래스가 입힌다(DESIGN.md §7.3). */
  *, *::before, *::after { box-sizing:border-box; }
  *:focus-visible { outline:var(--stroke-focus) solid var(--border-focus); outline-offset:1px; }
  body { min-height:100vh; display:grid; place-items:center; color:var(--ink); font-family:var(--font-sans); }
  main.pf-card { width:min(400px,calc(100vw - 32px)); padding:var(--space-8); }
  main h1 { margin:var(--space-2) 0 var(--space-1); }
  main > p { margin:0 0 var(--space-6); }
  label { display:block; margin:var(--space-4) 0 var(--space-1); }
  .pf-control { width:100%; }
  .submit { width:100%; margin-top:var(--space-6); }
  .register-link { display:block; margin:var(--space-4) auto 0; }
  .login-foot { display:flex; justify-content:center; margin-top:var(--space-6); }
  .pf-scrim { display:grid; place-items:center; z-index:10; }
  .pf-modal { width:min(420px,calc(100vw - 32px)); }
  .register-modal { max-height:calc(100vh - 32px); overflow:auto; }
  .account-type { display:flex; margin:0 0 var(--space-2); }
  .account-type .pf-seg__item { flex:1; }
  .message { min-height:20px; margin:var(--space-2) 0 0; }
</style></head><body><main class="pf-card">
  <div class="label-12 pf-muted">LS JUMP-UP · 설비 모니터링</div>
  <h1 class="title-16">계정 로그인</h1><p class="body-13 pf-secondary">등록된 계정으로 대시보드에 접속합니다.</p>
  <form method="post" action="{{ url_for('dashboard_login') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <label for="login_id" class="label-12 pf-secondary">아이디</label><input id="login_id" name="login_id" class="pf-control pf-control--lg" maxlength="80" autocomplete="username" required autofocus>
    <label for="password" class="label-12 pf-secondary">비밀번호</label><input id="password" name="password" type="password" class="pf-control pf-control--lg" autocomplete="current-password" required>
    <button type="submit" class="pf-btn pf-btn--primary pf-btn--lg body-14 submit">로그인</button>
  </form>
  <button type="button" class="pf-btn pf-btn--ghost label-12 register-link" onclick="openRegister()">계정 생성</button>
  <div class="login-foot"><span class="pf-chip-synthetic micro-11">합성 데이터 · 교육용</span></div>
</main>
{% if error %}<div class="pf-scrim" id="login-error-modal" role="dialog" aria-modal="true"
 onclick="if(event.target===this){closeModal('login-error-modal')}"><div class="pf-modal">
  <p class="pf-notice pf-notice--critical pf-modal__body body-14" role="alert">{{ error }}</p>
  <div class="pf-modal__actions"><button type="button" class="pf-btn pf-btn--primary label-12" onclick="closeModal('login-error-modal')">확인</button></div>
</div></div>{% endif %}
<div class="pf-scrim" id="register-modal" role="dialog" aria-modal="true" style="display:{% if register_open %}grid{% else %}none{% endif %};"
 onclick="if(event.target===this){closeRegister()}"><div class="pf-modal register-modal">
  <h2 class="pf-modal__title title-16">계정 생성</h2>
  <div class="account-type pf-seg" role="group" aria-label="계정 종류"><button type="button" id="user-type" class="pf-seg__item label-12" aria-pressed="true" onclick="chooseRole('USER')">일반 계정 생성</button>
    <button type="button" id="admin-type" class="pf-seg__item label-12" aria-pressed="false" onclick="chooseRole('ADMIN')">관리자 계정 생성</button></div>
  <form method="post" action="{{ url_for('dashboard_register') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}"><input type="hidden" id="role" name="role" value="USER">
    <label for="register_name" class="label-12 pf-secondary">이름</label><input id="register_name" name="name" class="pf-control" maxlength="120" required>
    <label for="register_email" class="label-12 pf-secondary">이메일</label><input id="register_email" name="email" type="email" class="pf-control" maxlength="160" required>
    <label for="register_phone" class="label-12 pf-secondary">전화번호</label><input id="register_phone" name="phone" class="pf-control" maxlength="40" required>
    <label for="register_id" class="label-12 pf-secondary">아이디</label><input id="register_id" name="login_id" class="pf-control" maxlength="80" autocomplete="username" required>
    <label for="register_password" class="label-12 pf-secondary">비밀번호</label><input id="register_password" name="password" type="password" class="pf-control" autocomplete="new-password" required>
    <label for="register_confirm" class="label-12 pf-secondary">비밀번호 확인</label><input id="register_confirm" name="password_confirm" type="password" class="pf-control" autocomplete="new-password" required>
    <div id="invite-wrap" style="display:none"><label for="invite_code" class="label-12 pf-secondary">관리인 코드</label><input id="invite_code" name="invite_code" type="password" class="pf-control" autocomplete="off"></div>
    <p class="message pf-field__error label-12" role="status">{{ register_message or '' }}</p>
    <div class="pf-modal__actions"><button class="pf-btn pf-btn--ghost label-12" type="button" onclick="closeRegister()">닫기</button>
      <button class="pf-btn pf-btn--primary label-12" type="submit">계정 생성</button></div>
  </form>
</div></div>
{% if register_success %}<div class="pf-scrim" id="register-success-modal" role="dialog" aria-modal="true"><div class="pf-modal">
  <p class="pf-modal__body body-14">계정이 생성되었습니다. 로그인해 주세요.</p>
  <div class="pf-modal__actions"><button type="button" class="pf-btn pf-btn--primary label-12" onclick="closeModal('register-success-modal')">확인</button></div>
</div></div>{% endif %}
<script>
function closeModal(id){const modal=document.getElementById(id);if(modal){modal.remove();}}
function openRegister(){document.getElementById('register-modal').style.display='grid';}
function closeRegister(){document.getElementById('register-modal').style.display='none';}
function chooseRole(role){document.getElementById('role').value=role;const admin=role==='ADMIN';
 document.getElementById('user-type').setAttribute('aria-pressed',String(!admin));document.getElementById('admin-type').setAttribute('aria-pressed',String(admin));
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
        if admin_id:
            try:
                identity = account_identity_for_session(str(admin_id))
            except Exception:
                # 계정 DB를 확인할 수 없으면 보호된 화면을 열어 주지 않는다.
                abort(503)
            if (identity is None or identity[1] != session.get("role")
                    or session.get("account_pk") != identity[0]):
                previous_role = str(session.get("role", "UNKNOWN"))
                session.clear()
                record_login_event(
                    "ACT_SESSION_REVOKED", actor_id=str(admin_id), actor_role=previous_role,
                    status="BLOCKED", block_reason="계정 삭제 또는 역할 변경",
                    source="dashboard_auth.require_admin_session",
                )
                if request.path.startswith(("/_dash", "/assets/")):
                    abort(401)
                if request.path in ("/login", "/register"):
                    return None
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
            LOGIN_HTML, design_css=DESIGN_CSS, csrf_token=session["login_csrf"], error=error,
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
