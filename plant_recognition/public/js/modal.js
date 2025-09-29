// Modern Modal System with Animations
class Modal {
  constructor() {
    this.activeModal = null;
  }

  // Show confirmation modal
  confirm(title, message, options = {}) {
    return new Promise((resolve) => {
      const modal = this.createModal({
        title: title,
        body: message,
        buttons: [
          {
            text: options.cancelText || 'Cancel',
            class: 'modal-button modal-button-secondary',
            onClick: () => {
              this.close();
              resolve(false);
            }
          },
          {
            text: options.confirmText || 'Confirm',
            class: `modal-button ${options.danger ? 'modal-button-danger' : 'modal-button-primary'}`,
            onClick: () => {
              this.close();
              resolve(true);
            }
          }
        ]
      });

      document.body.appendChild(modal);
      this.activeModal = modal;
    });
  }

  // Show alert modal
  alert(title, message, options = {}) {
    return new Promise((resolve) => {
      const modal = this.createModal({
        title: title,
        body: message,
        buttons: [
          {
            text: options.buttonText || 'OK',
            class: 'modal-button modal-button-primary',
            onClick: () => {
              this.close();
              resolve();
            }
          }
        ]
      });

      document.body.appendChild(modal);
      this.activeModal = modal;
    });
  }

  // Show custom modal
  custom(config) {
    const modal = this.createModal(config);
    document.body.appendChild(modal);
    this.activeModal = modal;
    return modal;
  }

  // Create modal element
  createModal(config) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';

    const content = document.createElement('div');
    content.className = 'modal-content';

    // Header
    if (config.title) {
      const header = document.createElement('div');
      header.className = 'modal-header';

      const title = document.createElement('h3');
      title.className = 'modal-title';
      title.textContent = config.title;

      const closeBtn = document.createElement('button');
      closeBtn.className = 'modal-close';
      closeBtn.innerHTML = '×';
      closeBtn.onclick = () => this.close();

      header.appendChild(title);
      header.appendChild(closeBtn);
      content.appendChild(header);
    }

    // Body
    if (config.body) {
      const body = document.createElement('div');
      body.className = 'modal-body';

      if (typeof config.body === 'string') {
        body.innerHTML = config.body;
      } else {
        body.appendChild(config.body);
      }

      content.appendChild(body);
    }

    // Footer with buttons
    if (config.buttons && config.buttons.length > 0) {
      const footer = document.createElement('div');
      footer.className = 'modal-footer';

      config.buttons.forEach(button => {
        const btn = document.createElement('button');
        btn.className = button.class;
        btn.textContent = button.text;
        btn.onclick = button.onClick;
        footer.appendChild(btn);
      });

      content.appendChild(footer);
    }

    overlay.appendChild(content);

    // Close on overlay click
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        this.close();
      }
    });

    // Close on Escape key
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        this.close();
        document.removeEventListener('keydown', handleEscape);
      }
    };
    document.addEventListener('keydown', handleEscape);

    return overlay;
  }

  // Close active modal
  close() {
    if (this.activeModal) {
      this.activeModal.style.opacity = '0';
      setTimeout(() => {
        if (this.activeModal && this.activeModal.parentNode) {
          this.activeModal.parentNode.removeChild(this.activeModal);
        }
        this.activeModal = null;
      }, 300);
    }
  }

  // Show loading modal
  loading(message = 'Loading...') {
    const loadingContent = document.createElement('div');
    loadingContent.style.cssText = `
      text-align: center;
      padding: 2rem;
    `;

    loadingContent.innerHTML = `
      <div class="loading-overlay" style="position: relative; height: 4rem; margin-bottom: 1rem;"></div>
      <p style="margin: 0; color: var(--text-secondary);">${message}</p>
    `;

    const modal = this.createModal({
      body: loadingContent
    });

    document.body.appendChild(modal);
    this.activeModal = modal;
    return modal;
  }

  // Show success message
  success(title, message) {
    return this.alert(title, `
      <div style="text-align: center; margin-bottom: 1rem;">
        <div style="width: 4rem; height: 4rem; background: var(--success); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; color: white; font-size: 1.5rem;">✓</div>
        <p style="margin: 0; color: var(--text);">${message}</p>
      </div>
    `);
  }

  // Show error message
  error(title, message) {
    return this.alert(title, `
      <div style="text-align: center; margin-bottom: 1rem;">
        <div style="width: 4rem; height: 4rem; background: var(--error); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; color: white; font-size: 1.5rem;">!</div>
        <p style="margin: 0; color: var(--text);">${message}</p>
      </div>
    `);
  }
}

// Global modal instance
window.modal = new Modal();

// Convenience functions for backward compatibility
window.showModal = {
  confirm: (title, message, options) => window.modal.confirm(title, message, options),
  alert: (title, message, options) => window.modal.alert(title, message, options),
  success: (title, message) => window.modal.success(title, message),
  error: (title, message) => window.modal.error(title, message),
  loading: (message) => window.modal.loading(message)
};