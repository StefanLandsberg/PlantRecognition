(() => {
  try {
    const html = document.documentElement;

    // Reset classes (optional if you want clean state each time)
    html.classList.remove('dark-mode', 'light-mode', 'protanopia', 'deuteranopia', 'tritanopia');

    // Dark mode
    const dark = localStorage.getItem('darkMode') !== 'false';
    html.classList.add(dark ? 'dark-mode' : 'light-mode');

    // Color blind mode
    const cb = localStorage.getItem('colorBlindMode');
    if (cb && cb !== 'none') html.classList.add(cb);
  } catch (e) {
    console.error('Theme init failed:', e);
  }
})();
