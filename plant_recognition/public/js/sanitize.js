export function sanitizeHtml(str='') {
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}
