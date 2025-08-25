(function () {
  const ready = (fn) =>
    document.readyState === "loading"
      ? document.addEventListener("DOMContentLoaded", fn)
      : fn();

  ready(() => {
    const menuBtn = document.getElementById("menu-btn");
    const menu = document.getElementById("menu-dropdown");

    if (menuBtn && menu) {
      menuBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        menu.classList.toggle("hidden");
        menu.setAttribute(
          "aria-hidden",
          menu.classList.contains("hidden") ? "true" : "false"
        );
      });

      document.addEventListener("click", (e) => {
        if (!menu.classList.contains("hidden")) {
          const inside = menu.contains(e.target) || menuBtn.contains(e.target);
          if (!inside) menu.classList.add("hidden");
        }
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") menu.classList.add("hidden");
      });
    }

    const logoutBtn = document.getElementById("btn-logout");
    if (logoutBtn) {
      logoutBtn.addEventListener("click", async () => {
        try {
          await fetch("/api/auth/logout", {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          });
        } catch {}
        location.href = "/";
      });
    }
  });
})();
