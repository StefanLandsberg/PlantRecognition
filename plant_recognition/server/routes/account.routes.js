// server/routes/account.routes.js
import express from "express";
import bcrypt from "bcryptjs";
import { requireAuth } from "../middleware/auth.js";
import User from "../models/User.js";

const router = express.Router();


router.put("/profile", requireAuth, async (req, res, next) => {
  try {
    const userId = req.auth.userId;
    const { name, username, email } = req.body;

    const update = {};
    if (typeof name === "string") update.username = name.trim();      
    if (typeof username === "string") update.username = username.trim();
    if (typeof email === "string") update.email = email.trim().toLowerCase();

    if (update.email) {
      const exists = await User.findOne({ _id: { $ne: userId }, email: update.email }).lean();
      if (exists) return res.status(409).json({ error: "Email already in use" });
    }
    if (update.username) {
      const exists = await User.findOne({ _id: { $ne: userId }, username: update.username }).lean();
      if (exists) return res.status(409).json({ error: "Username already in use" });
    }

    await User.findByIdAndUpdate(userId, update, { runValidators: true });
    return res.json({ ok: true });
  } catch (err) {
    next(err);
  }
});


router.put("/password", requireAuth, async (req, res, next) => {
  try {
    const userId = req.auth.userId;
    const { currentPassword, newPassword } = req.body;
    if (!currentPassword || !newPassword) {
      return res.status(400).json({ error: "Missing password fields" });
    }

    const user = await User.findById(userId).select("+password");
    if (!user) return res.status(404).json({ error: "User not found" });

    const ok = await bcrypt.compare(currentPassword, user.password || "");
    if (!ok) return res.status(400).json({ error: "Current password is incorrect" });

    const hash = await bcrypt.hash(newPassword, 10);
    user.password = hash;
    await user.save();

    return res.json({ ok: true });
  } catch (err) {
    next(err);
  }
});


router.delete("/", requireAuth, async (req, res, next) => {
  try {
    const userId = req.auth.userId;
    await User.findByIdAndDelete(userId);
    return res.json({ ok: true });
  } catch (err) {
    next(err);
  }
});

export default router;
