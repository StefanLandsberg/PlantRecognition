import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateRecord, manifestSchema } from "../utils/manifestRules.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const MANIFEST_JSON = path.join(PROJECT_ROOT, "data", "manifest.json");

export async function getManifest(req, res, next) {
  try {
    if (!fs.existsSync(MANIFEST_JSON))
      return res.status(404).json({ error: "manifest.json not found" });

    let data = JSON.parse(fs.readFileSync(MANIFEST_JSON, "utf-8"));
    const { invasive } = req.query;
    if (invasive === "true") data = data.filter(r => r.is_invasive);
    if (invasive === "false") data = data.filter(r => !r.is_invasive);

    res.json({ count: data.length, schema: manifestSchema, data });
  } catch (e) { next(e); }
}

export async function validateData(req, res, next) {
  try {
    const payload = Array.isArray(req.body) ? req.body : [req.body];
    res.json({ results: payload.map(r => validateRecord(r)) });
  } catch (e) { next(e); }
}
