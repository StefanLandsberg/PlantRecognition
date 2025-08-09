export function pickFile(inputId, cb) {
  const input = document.getElementById(inputId);
  input.onchange = () => {
    const f = input.files?.[0];
    if (f) cb(f);
    input.value = '';
  };
  input.click();
}
