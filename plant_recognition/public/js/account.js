// /public/js/account.js
import { AuthAPI } from "./api.js";

function setLoading(btn, on) {
  if (!btn) return;
  btn.disabled = !!on;
  btn.classList.toggle("is-loading", !!on);
  btn.classList.toggle("loading-overlay", !!on);
  if (on) {
    btn.dataset._label = btn.textContent;
    btn.textContent = "Processing...";
  } else if (btn.dataset._label) {
    btn.textContent = btn.dataset._label;
    btn.classList.remove("loading-overlay");
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
    await showModal.success("Profile Updated", "Your profile has been saved successfully.");
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
    await showModal.success("Password Updated", "Your password has been changed successfully.");
  } catch (err) {
    showErr("pwd-error", typeof err?.message === "string" ? err.message : "Failed to update password.");
  } finally {
    setLoading(btn, false);
  }
}

async function onDeleteAccount(e) {
  const btn = e.currentTarget;
  try {
    const confirmed = await showModal.confirm(
      "Delete Account",
      "This will permanently delete your account and all your data. This action cannot be undone.",
      {
        danger: true,
        confirmText: "Delete Account",
        cancelText: "Cancel"
      }
    );

    if (!confirmed) return;

    setLoading(btn, true);
    const res = await fetch("/api/account", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: "{}",
    });
    if (!res.ok) throw new Error(await res.text());

    await showModal.success("Account Deleted", "Your account has been permanently deleted.");
    location.href = "/";
  } catch (err) {
    await showModal.error("Error", typeof err?.message === "string" ? err.message : "Failed to delete account.");
  } finally {
    setLoading(btn, false);
  }
}

document.addEventListener("DOMContentLoaded", boot);

// Use the reusable password validation function from auth.js
// Load auth.js functions if not already available
if (typeof setupPasswordValidation === 'undefined') {
    // Import the function from auth.js or define it here if needed
    // For now, use a simplified inline version until proper module loading is set up
    const script = document.createElement('script');
    script.src = '/js/auth.js';
    document.head.appendChild(script);
    script.onload = () => {
        setupPasswordValidation('pwd-new');
    };
} else {
    setupPasswordValidation('pwd-new');
}
