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
          if (!inside) {
            menu.classList.add("hidden");
            menu.setAttribute("aria-hidden", "true");
          }
        }
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          menu.classList.add("hidden");
          menu.setAttribute("aria-hidden", "true");
        }
      });

      // Close menu when any navigation link is clicked
      const navLinks = menu.querySelectorAll("a");
      navLinks.forEach(link => {
        link.addEventListener("click", () => {
          menu.classList.add("hidden");
          menu.setAttribute("aria-hidden", "true");
        });
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

    // Mobile QR Code button functionality
    const navMobileQR = document.getElementById("nav-mobile-qr");
    console.log('Setting up Mobile QR button, element found:', !!navMobileQR);
    if (navMobileQR) {
      console.log('Adding click listener to Mobile QR button');
      navMobileQR.addEventListener("click", async (event) => {
        console.log('Mobile QR button clicked!');
        event.preventDefault();
        try {
          console.log('Requesting QR code...');
          const response = await fetch('/qr-mobile');
          const data = await response.json();
          console.log('QR response:', data);

          if (data.success) {
            console.log('Calling showQRModal with:', {
              hasQRCode: !!data.qrCode,
              mobileUrl: data.mobileUrl,
              companionCode: data.companionCode
            });
            showQRModal(data.qrCode, data.mobileUrl, data.companionCode);
          } else {
            console.error('QR generation failed:', data.error);
            alert('Failed to generate QR code: ' + data.error);
          }
        } catch (error) {
          console.error('QR code generation error:', error);
          alert('Failed to generate QR code');
        }
      });
    }

    function showQRModal(qrCodeDataURL, mobileUrl, companionCode) {
      // Remove existing modal if any
      const existingModal = document.getElementById('qr-modal');
      if (existingModal) {
        existingModal.remove();
      }

      // Debug: Log the parameters to see what we're receiving
      console.log('showQRModal called with:', { qrCodeDataURL: !!qrCodeDataURL, mobileUrl, companionCode });

      // Ensure companionCode is defined
      if (!companionCode) {
        console.error('Missing companion code');
        alert('Failed to generate companion code');
        return;
      }

      // Create modal
      const modal = document.createElement('div');
      modal.id = 'qr-modal';
      modal.className = 'qr-modal';

      const modalContent = document.createElement('div');
      modalContent.className = 'qr-modal-content';

      modalContent.innerHTML = `
        <h3 class="qr-modal-title">Scan for Mobile Access</h3>
        <img src="${qrCodeDataURL}" alt="QR Code" class="qr-code-image">
        <p class="qr-modal-subtitle">Scan with your phone's camera app</p>
        <div class="qr-modal-buttons">
          <button id="copy-url-btn" class="qr-modal-btn primary">Copy URL</button>
          <button id="close-qr-modal" class="qr-modal-btn secondary">Close</button>
        </div>
      `;

      modal.appendChild(modalContent);
      document.body.appendChild(modal);

      // Event listeners
      modalContent.querySelector('#close-qr-modal').addEventListener('click', () => {
        modal.remove();
      });

      modalContent.querySelector('#copy-url-btn').addEventListener('click', () => {
        navigator.clipboard.writeText(mobileUrl).then(() => {
          const btn = modalContent.querySelector('#copy-url-btn');
          btn.textContent = 'Copied!';
          setTimeout(() => {
            btn.textContent = 'Copy URL';
          }, 2000);
        }).catch(() => {
          alert('Failed to copy URL');
        });
      });

      // Close on backdrop click
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.remove();
        }
      });
    }
  });
})();
