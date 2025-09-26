import express from "express";
import { getManifest, validateData } from "../controllers/data.controller.js";
import { requireAuth } from "../middleware/auth.js";

const router = express.Router();

router.get("/manifest", requireAuth, getManifest);
router.post("/validate", requireAuth, validateData);

export default router;
