// /public/js/account.js
import { AuthAPI } from "./api.js";

function setLoading(btn, on) {
  if (!btn) return;
  btn.disabled = !!on;
  btn.classList.toggle("is-loading", !!on);
  if (on) {
    btn.dataset._label = btn.textContent;
    btn.textContent = "Please wait…";
  } else if (btn.dataset._label) {
    btn.textContent = btn.dataset._label;
  }
}

function showErr(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg || "";
}

let _profile = { name: "", email: "" };

async function boot() {
  try {
    const me = await AuthAPI.me();
    _profile = { name: me?.name || "", email: me?.email || "" };
    document.getElementById("acc-name").value = _profile.name;
    document.getElementById("acc-email").value = _profile.email;
  } catch {
    location.href = "/";
    return;
  }

  document.getElementById("acc-save")?.addEventListener("click", onSaveProfile);
  document.getElementById("acc-cancel")?.addEventListener("click", onResetProfile);
  document.getElementById("pwd-update")?.addEventListener("click", onUpdatePassword);
  document.getElementById("acc-delete")?.addEventListener("click", onDeleteAccount);
}

function onResetProfile() {
  document.getElementById("acc-name").value = _profile.name;
  document.getElementById("acc-email").value = _profile.email;
  showErr("acc-error", "");
}

function isValidEmail(v) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
}

async function onSaveProfile(e) {
  const btn = e.currentTarget;
  showErr("acc-error", "");
  try {
    const name = document.getElementById("acc-name").value.trim();
    const email = document.getElementById("acc-email").value.trim();

    if (!name) throw new Error("Please enter your display name.");
    if (!email || !isValidEmail(email)) throw new Error("Please enter a valid email address.");

    setLoading(btn, true);
    const res = await fetch("/api/account/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ name, email }),
    });
    if (!res.ok) throw new Error(await res.text());
    _profile = { name, email };
    alert("Profile saved");
  } catch (err) {
    showErr("acc-error", typeof err?.message === "string" ? err.message : "Failed to save profile.");
  } finally {
    setLoading(btn, false);
  }
}

async function onUpdatePassword(e) {
  const btn = e.currentTarget;
  showErr("pwd-error", "");
  try {
    const cur = document.getElementById("pwd-current").value;
    const next = document.getElementById("pwd-new").value;
    const conf = document.getElementById("pwd-confirm").value;

    if (!cur || !next || !conf) throw new Error("Please fill in all password fields.");
    if (next.length < 8) throw new Error("Use at least 8 characters for your new password.");
    if (next !== conf) throw new Error("New passwords do not match.");

    setLoading(btn, true);
    const res = await fetch("/api/account/password", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ currentPassword: cur, newPassword: next }),
    });
    if (!res.ok) throw new Error(await res.text());

    ["pwd-current","pwd-new","pwd-confirm"].forEach(id => (document.getElementById(id).value = ""));
    alert("Password updated");
  } catch (err) {
    showErr("pwd-error", typeof err?.message === "string" ? err.message : "Failed to update password.");
  } finally {
    setLoading(btn, false);
  }
}

async function onDeleteAccount(e) {
  const btn = e.currentTarget;
  try {
    const ok = confirm("This will permanently delete your account and data. Continue?");
    if (!ok) return;

    setLoading(btn, true);
    const res = await fetch("/api/account", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: "{}",
    });
    if (!res.ok) throw new Error(await res.text());

    alert("Account deleted");
    location.href = "/";
  } catch (err) {
    alert(typeof err?.message === "string" ? err.message : "Failed to delete account.");
  } finally {
    setLoading(btn, false);
  }
}

document.addEventListener("DOMContentLoaded", boot);

//code for password recommendations
// Get a reference to the input field and all requirement elements
const passwordInput = document.getElementById('pwd-new');
const lengthRequirement = document.getElementById('length');
const uppercaseRequirement = document.getElementById('uppercase');
const numberRequirement = document.getElementById('number');
const specialRequirement = document.getElementById('special');

// Add an event listener for the 'keyup' event
passwordInput.addEventListener('keyup', () => {
    const password = passwordInput.value;

    // 1. Check for password length
    if (password.length >= 8) {
        lengthRequirement.classList.remove('invalid');
        lengthRequirement.classList.add('valid');
    } else {
        lengthRequirement.classList.remove('valid');
        lengthRequirement.classList.add('invalid');
    }

    // 2. Check for at least one uppercase letter (A-Z)
    if (/[A-Z]/.test(password)) {
        uppercaseRequirement.classList.remove('invalid');
        uppercaseRequirement.classList.add('valid');
    } else {
        uppercaseRequirement.classList.remove('valid');
        uppercaseRequirement.classList.add('invalid');
    }

    // 3. Check for at least one number (0-9)
    if (/\d/.test(password)) {
        numberRequirement.classList.remove('invalid');
        numberRequirement.classList.add('valid');
    } else {
        numberRequirement.classList.remove('valid');
        numberRequirement.classList.add('invalid');
    }

    // 4. Check for at least one special character
    // This regular expression matches common special characters
    if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]/.test(password)) {
        specialRequirement.classList.remove('invalid');
        specialRequirement.classList.add('valid');
    } else {
        specialRequirement.classList.remove('valid');
        specialRequirement.classList.add('invalid');
    }
});
