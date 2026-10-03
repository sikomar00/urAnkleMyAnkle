/* Dash 기본 컴포넌트(dcc.Dropdown = react-select v1, dcc.Slider = rc-slider, DataTable)는
   접근 이름·역할을 받는 속성을 Python 쪽에 노출하지 않는다. 렌더된 DOM에 보충하고,
   화면이 다시 그려질 때마다 MutationObserver로 다시 적용한다(속성 변경은 관찰하지 않아
   스스로를 다시 부르지 않는다). */
(function () {
  // 라벨 요소가 옆에 없는 드롭다운의 이름. 필터바 드롭다운은 바로 앞 라벨 글자를 쓴다.
  var DROPDOWN_NAMES = { "export-dd": "보고서 형식과 대상" };
  // dcc.Input은 aria-* 속성을 받지 않아 Python 쪽에서 이름을 줄 수 없다.
  var INPUT_NAMES = { "start-date": "시작일", "end-date": "종료일" };
  var PAGER_NAMES = {
    "first-page": "첫 페이지", "previous-page": "이전 페이지",
    "next-page": "다음 페이지", "last-page": "마지막 페이지"
  };

  function dropdownName(dropdown) {
    var previous = dropdown.previousElementSibling;
    if (previous && previous.textContent.trim()) { return previous.textContent.trim(); }
    return DROPDOWN_NAMES[dropdown.id] || null;
  }

  function nearestLabel(node) {
    while (node && node !== document.body) {
      var label = node.querySelector("label");
      if (label && label.textContent.trim()) { return label.textContent.trim(); }
      node = node.parentElement;
    }
    return null;
  }

  function apply() {
    document.querySelectorAll(".dash-dropdown[id]").forEach(function (dropdown) {
      var name = dropdownName(dropdown);
      if (name) {
        dropdown.querySelectorAll(".Select-input, .Select-input > input").forEach(function (el) {
          if (el.getAttribute("aria-label") !== name) { el.setAttribute("aria-label", name); }
        });
      }
      // 비활성 상태를 보조기술에도 알린다(react-select v1은 클래스로만 표시한다).
      var select = dropdown.querySelector(".Select");
      if (select) {
        if (select.classList.contains("is-disabled")) { select.setAttribute("aria-disabled", "true"); }
        else { select.removeAttribute("aria-disabled"); }
      }
    });
    // 선택된 값 표시에 붙은 role="option"·aria-selected는 listbox 밖이라 잘못된 역할이다.
    document.querySelectorAll('.Select-value-label[role="option"]').forEach(function (el) {
      el.removeAttribute("role");
      el.removeAttribute("aria-selected");
    });
    Object.keys(INPUT_NAMES).forEach(function (id) {
      var el = document.getElementById(id);
      if (el && el.getAttribute("aria-label") !== INPUT_NAMES[id]) {
        el.setAttribute("aria-label", INPUT_NAMES[id]);
      }
    });
    document.querySelectorAll(".rc-slider-handle:not([aria-label])").forEach(function (el) {
      el.setAttribute("aria-label", nearestLabel(el.closest(".rc-slider")) || "값 조절");
    });
    document.querySelectorAll(".previous-next-container button").forEach(function (button) {
      Object.keys(PAGER_NAMES).forEach(function (cls) {
        if (button.classList.contains(cls) && !button.getAttribute("aria-label")) {
          button.setAttribute("aria-label", PAGER_NAMES[cls]);
        }
      });
    });
    // dcc.Tabs의 탭은 초점을 받지 않는 div다 — 키보드로 화면을 바꿀 수 있게 탭 역할과 초점을 준다.
    document.querySelectorAll("#screen-tabs").forEach(function (list) { list.setAttribute("role", "tablist"); });
    document.querySelectorAll("#screen-tabs .tab").forEach(function (tab) {
      var selected = tab.classList.contains("tab--selected") ? "true" : "false";
      if (tab.getAttribute("role") !== "tab") { tab.setAttribute("role", "tab"); }
      if (tab.getAttribute("tabindex") !== "0") { tab.setAttribute("tabindex", "0"); }
      if (tab.getAttribute("aria-selected") !== selected) { tab.setAttribute("aria-selected", selected); }
      if (!tab.dataset.keyboard) {
        tab.dataset.keyboard = "1";
        tab.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); tab.click(); }
        });
      }
    });
    // DataTable 열 정렬 — 정렬 아이콘 span은 초점을 받지 않는다. 버튼 역할·이름·초점을 주고 Enter·Space로
    // 누르게 하며, 현재 방향을 th의 aria-sort로 알린다(다른 표의 정렬 버튼과 같은 순환: 오름 → 내림 → 원래).
    document.querySelectorAll(".pf-dtable th.dash-header").forEach(function (th) {
      var sortEl = th.querySelector(".column-header--sort");
      var nameEl = th.querySelector(".column-header-name");
      if (!sortEl || !nameEl) { return; }
      var icon = sortEl.querySelector("svg");
      var state = icon ? icon.getAttribute("data-icon") : "sort";
      var sort = state === "sort-up" ? "ascending" : state === "sort-down" ? "descending" : null;
      var action = sort === "ascending" ? "내림차순으로 정렬" : sort === "descending" ? "원래 순서로 되돌림" : "오름차순으로 정렬";
      var label = nameEl.textContent.trim() + ", " + action;
      if (sort) { if (th.getAttribute("aria-sort") !== sort) { th.setAttribute("aria-sort", sort); } }
      else if (th.hasAttribute("aria-sort")) { th.removeAttribute("aria-sort"); }
      if (sortEl.getAttribute("role") !== "button") { sortEl.setAttribute("role", "button"); }
      if (sortEl.getAttribute("tabindex") !== "0") { sortEl.setAttribute("tabindex", "0"); }
      if (sortEl.getAttribute("aria-label") !== label) { sortEl.setAttribute("aria-label", label); }
      if (th.getAttribute("title") !== label) { th.setAttribute("title", label); }
      if (!sortEl.dataset.keyboard) {
        sortEl.dataset.keyboard = "1";
        sortEl.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); sortEl.click(); }
        });
      }
    });
    // 가로 스크롤 표는 키보드로도 스크롤할 수 있어야 한다.
    document.querySelectorAll(".dash-spreadsheet-container:not([tabindex])").forEach(function (el) {
      el.setAttribute("tabindex", "0");
      el.setAttribute("role", "region");
      el.setAttribute("aria-label", "데이터 조회 표");
    });
  }

  var scheduled = false;
  new MutationObserver(function () {
    if (scheduled) { return; }
    scheduled = true;
    window.requestAnimationFrame(function () { scheduled = false; apply(); });
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
