// /js/account/handlers.js
import { AuthAPI } from "../api.js";
import { setLoading, showErr } from "./ui.js";
import { isValidEmail } from "./utils.js";

// Page-level state for the user's profile
let _profile = { name: "", email: "" };

const onResetProfile = () => {
  document.getElementById("acc-name").value = _profile.name;
  document.getElementById("acc-email").value = _profile.email;
  showErr("acc-error", "");
};

const onSaveProfile = async (e) => {
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
    showErr("acc-error", err instanceof Error ? err.message : "Failed to save profile.");
  } finally {
    setLoading(btn, false);
  }
};

const onUpdatePassword = async (e) => {
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

    ["pwd-current", "pwd-new", "pwd-confirm"].forEach(id => (document.getElementById(id).value = ""));
    alert("Password updated");
  } catch (err) {
    showErr("pwd-error", err instanceof Error ? err.message : "Failed to update password.");
  } finally {
    setLoading(btn, false);
  }
};

const onDeleteAccount = async (e) => {
  const btn = e.currentTarget;
  try {
    if (!confirm("This will permanently delete your account and data. Continue?")) {
      return;
    }

    setLoading(btn, true);
    const res = await fetch("/api/account", {
      method: "DELETE",
      credentials: "include",
    });
    if (!res.ok) throw new Error(await res.text());

    alert("Account deleted");
    location.href = "/";
  } catch (err) {
    alert(err instanceof Error ? err.message : "Failed to delete account.");
    setLoading(btn, false);
  }
};

/**
 * Initializes the account page by fetching user data and attaching event listeners.
 */
export const initializeAccountPage = async () => {
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
};