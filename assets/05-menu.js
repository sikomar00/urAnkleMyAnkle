/* 계정 메뉴(<details>)에서 모달을 여는 버튼(비밀번호 변경·계정 관리·로그아웃)을 누르면 메뉴를 닫는다.
   메뉴와 모달이 함께 떠 있으면 그림자가 둘이 된다(DESIGN.md "화면에 동시에 떠 있는 그림자는 하나다").
   "연장"은 메뉴 안에서 끝나는 동작이라 메뉴를 그대로 둔다. 화면이 다시 그려져도 동작하도록
   document에 위임한다. */
document.addEventListener("click", function (event) {
  var button = event.target.closest(".pf-menu .pf-btn");
  if (!button || button.id === "session-extend-btn") { return; }
  var menu = button.closest("details");
  if (menu) { menu.open = false; }
});
