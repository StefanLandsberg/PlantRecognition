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

    // Companion Code Management
    const companionCodeElement = document.getElementById("companion-code");
    if (companionCodeElement) {
      let companionCode = localStorage.getItem('companionCode');

      // Generate new code if none exists or if it's older than 24 hours
      if (!companionCode || isCodeExpired()) {
        companionCode = generateCompanionCode();
        localStorage.setItem('companionCode', companionCode);
        localStorage.setItem('companionCodeTimestamp', Date.now().toString());
      }

      companionCodeElement.textContent = companionCode;

      // Always register the code with the server (to ensure it's active)
      registerCompanionCode(companionCode);

      // Click to copy functionality
      companionCodeElement.addEventListener('click', () => {
        navigator.clipboard.writeText(companionCode).then(() => {
          const originalText = companionCodeElement.textContent;
          companionCodeElement.textContent = 'Copied!';
          setTimeout(() => {
            companionCodeElement.textContent = originalText;
          }, 1000);
        }).catch(() => {
          // Fallback for older browsers
          const textArea = document.createElement('textarea');
          textArea.value = companionCode;
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand('copy');
          document.body.removeChild(textArea);

          const originalText = companionCodeElement.textContent;
          companionCodeElement.textContent = 'Copied!';
          setTimeout(() => {
            companionCodeElement.textContent = originalText;
          }, 1000);
        });
      });
    }

    function generateCompanionCode() {
      return Math.floor(100000 + Math.random() * 900000).toString();
    }

    function isCodeExpired() {
      const timestamp = localStorage.getItem('companionCodeTimestamp');
      if (!timestamp) return true;

      const codeAge = Date.now() - parseInt(timestamp);
      const twentyFourHours = 24 * 60 * 60 * 1000;
      return codeAge > twentyFourHours;
    }

    async function registerCompanionCode(code) {
      try {
        console.log('Registering companion code:', code);
        const response = await fetch('/api/companion/register', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          credentials: 'include',
          body: JSON.stringify({ companionCode: code })
        });

        const result = await response.json();
        console.log('Registration result:', result);

        if (!response.ok) {
          console.error('Registration failed:', result);
        }
      } catch (error) {
        console.error('Failed to register companion code:', error);
      }
    }
  });
})();
