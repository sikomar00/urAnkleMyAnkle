"""한 명의 관리자 계정으로 Dash 접근을 통제하는 Flask 로그인 화면."""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import timedelta

from flask import abort, redirect, render_template_string, request, session, url_for

from .audit_service import log_failure, record_action, verify_admin


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
  .error { margin:15px 0 0; padding:10px; border-radius:7px; background:#fff0ed; color:#9e3727; }
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
  {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
  <small>관리자 1명 전용 · 회원가입 없음</small>
</main></body></html>"""


def install_auth(server) -> None:
    """인증되지 않은 요청이 Dash 데이터 콜백에 접근하지 못하게 한다."""
    # 로컬 개발에서 .env가 빠져도 예측하기 어려운 임시 키를 사용한다.
    # 운영·재시작 간 세션 유지에는 .env의 고정된 무작위 키가 필요하다.
    server.secret_key = os.environ.get("DASHBOARD_SESSION_SECRET") or secrets.token_urlsafe(48)
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("DASHBOARD_COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    )

    @server.after_request
    def prevent_private_page_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @server.before_request
    def require_admin_session():
        if request.path == "/login":
            return None
        if session.get("admin_id"):
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
                    session.clear()
                    session.permanent = True
                    session["admin_id"] = login_id
                    record_action("ACT_LOGIN", "관리자 로그인", target_type="admin",
                                  target_id=login_id)
                    return redirect("/")
                record_action("ACT_LOGIN", "관리자 로그인", status="BLOCKED",
                              block_reason="아이디 또는 비밀번호 불일치")
                error, status = "아이디 또는 비밀번호가 올바르지 않습니다.", 401
        return render_template_string(LOGIN_HTML, csrf_token=session["login_csrf"], error=error), status
