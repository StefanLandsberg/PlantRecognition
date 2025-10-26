import Sighting from "../models/Sighting.js";
import VideoSession from "../models/VideoSession.js";
import User from "../models/User.js";
import { saveFile } from "../services/storage.service.js";
import { queueLLMAnalysis } from "../services/llm.service.js";
import { publish } from "../services/sse.service.js";
import { PythonShell } from "python-shell";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import mongoose from "mongoose";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use same path resolution as upload middleware
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, "uploads");

// Duplicate detection configuration
const DUPLICATE_CONFIG = {
  RADIUS_METERS: 15, // Detection radius in meters
  TIME_WINDOW_DAYS: 10, // How many days to look back
  MIN_PREVIOUS_CONFIDENCE: 0, // Minimum confidence for previous detection (same as current)
  MIN_CURRENT_CONFIDENCE: 0, // Minimum confidence for current detection to trigger check
};

function deleteStoredUpload(imageUrlPath) {
  if (!imageUrlPath) return;
  try {
    const storedFile = path.join(UPLOADS_DIR, path.basename(imageUrlPath));
    if (fs.existsSync(storedFile)) {
      fs.unlinkSync(storedFile);
      console.log(`[DUPLICATE] Removed stored upload ${storedFile}`);
    }
  } catch (error) {
    console.warn(
      `[DUPLICATE] Failed to clean stored upload ${imageUrlPath}:`,
      error
    );
  }
}

function scheduleUploadCleanup(imageUrlPath, delayMs = 60_000) {
  if (!imageUrlPath) return;
  const timer = setTimeout(() => deleteStoredUpload(imageUrlPath), delayMs);
  if (typeof timer.unref === "function") {
    timer.unref(); // don't keep the event loop alive just for cleanup
  }
}

async function checkForDuplicate(
  species,
  confidence,
  lat,
  lng,
  userId,
  options = {}
) {
  const { excludeId = null } = options;
  const normalizedExcludeId = normalizeObjectId(excludeId);
  console.log(
    `[DUPLICATE CHECK] Starting check for species: "${species}", confidence: ${confidence}, lat: ${lat}, lng: ${lng}, userId: ${userId}`
  );

  // Skip duplicate check if confidence too low or coordinates invalid
  if (
    !species ||
    confidence < DUPLICATE_CONFIG.MIN_CURRENT_CONFIDENCE ||
    !lat ||
    !lng
  ) {
    return null;
  }

  try {
    const timeThreshold = new Date(
      Date.now() - DUPLICATE_CONFIG.TIME_WINDOW_DAYS * 24 * 60 * 60 * 1000
    );
    const baseFilters = {
      owner: userId,
      "analysis.predictedSpecies": species,
      "analysis.confidence": {
        $gte: DUPLICATE_CONFIG.MIN_PREVIOUS_CONFIDENCE,
      },
      createdAt: { $gte: timeThreshold },
      isRemoved: { $ne: true },
    };

    if (normalizedExcludeId) {
      baseFilters._id = { $ne: normalizedExcludeId };
    }

    // Try geospatial query first - use exact string match
    try {
      const nearbyDetection = await Sighting.findOne({
        ...baseFilters,
        location: {
          $near: {
            $geometry: {
              type: "Point",
              coordinates: [parseFloat(lng), parseFloat(lat)],
            },
            $maxDistance: DUPLICATE_CONFIG.RADIUS_METERS,
          },
        },
      }).sort({ createdAt: -1 });

      if (nearbyDetection) {
        console.log(
          `[DUPLICATE CHECK] Found geospatial duplicate:`,
          nearbyDetection._id
        );
        return nearbyDetection;
      } else {
        console.log(`[DUPLICATE CHECK] No geospatial duplicates found`);
      }
    } catch (geoError) {
      console.warn(
        "Geospatial query failed, falling back to manual distance calculation:",
        geoError
      );
    }

    // Fallback: manual distance calculation
    const radiusInDegrees = DUPLICATE_CONFIG.RADIUS_METERS / 111000; // Approximate meters to degrees
    const recentDetections = await Sighting.find({
      ...baseFilters,
      location: { $exists: true, $ne: null },
    }).sort({ createdAt: -1 });

    console.log(
      `[DUPLICATE CHECK] Found ${recentDetections.length} recent detections for fallback check`
    );

    // Check each detection manually
    for (const detection of recentDetections) {
      if (detection.location && detection.location.coordinates) {
        if (
          normalizedExcludeId &&
          detection._id &&
          typeof detection._id.equals === "function" &&
          detection._id.equals(normalizedExcludeId)
        ) {
          continue;
        }
        const [detLng, detLat] = detection.location.coordinates;
        const latDiff = Math.abs(detLat - parseFloat(lat));
        const lngDiff = Math.abs(detLng - parseFloat(lng));

        // Simple bounding box check (approximate)
        if (latDiff <= radiusInDegrees && lngDiff <= radiusInDegrees) {
          // More precise distance calculation
          const distance = calculateDistance(
            parseFloat(lat),
            parseFloat(lng),
            detLat,
            detLng
          );
          if (distance <= DUPLICATE_CONFIG.RADIUS_METERS) {
            console.log(
              `[DUPLICATE CHECK] Found manual duplicate: distance=${distance}m, detection=${detection._id}`
            );
            return detection;
          }
        }
      }
    }

    console.log(`[DUPLICATE CHECK] No duplicates found after all checks`);
    return null;
  } catch (error) {
    console.error("[DUPLICATE CHECK] Error checking for duplicates:", error);
    return null; // On error, allow the detection (fail safe)
  }
}

// Haversine formula to calculate distance between two points
function calculateDistance(lat1, lng1, lat2, lng2) {
  const R = 6371000; // Earth's radius in meters
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c; // Distance in meters
}

function normalizeObjectId(value) {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (value instanceof Buffer) return value.toString();
  if (typeof value === "object") {
    if (value._id) return value._id.toString();
    if (typeof value.toString === "function") {
      const str = value.toString();
      if (!str.startsWith("[object")) return str;
    }
  }
  return null;
}

export async function analyzeOnce(req, res, next) {
  try {
    const { file } = req;
    if (!file) return res.status(400).json({ error: "Image required" });

    const {
      lat,
      lng,
      videoSessionId: rawVideoSessionId,
      videoTimestamp: rawVideoTimestamp,
      localFileId,
      storageType,
    } = req.body;

    const normalizedVideoSessionId =
      rawVideoSessionId &&
      typeof rawVideoSessionId === "string" &&
      mongoose.Types.ObjectId.isValid(rawVideoSessionId)
        ? rawVideoSessionId
        : null;
    const parsedVideoTimestamp = Number(rawVideoTimestamp);
    const normalizedVideoTimestamp = Number.isFinite(parsedVideoTimestamp)
      ? parsedVideoTimestamp
      : null;

    // Get user storage preference (set by storage-aware upload middleware)
    const storagePreference = req.userStoragePreference || "server";

    // Handle storage based on user preference
    const imageUrlPath = await saveFile(
      file,
      req.auth.userId,
      storagePreference
    );

    // Convert URL path to full filesystem path for ML model
    // For local storage users, still use the temp file for ML analysis
    const fullImagePath = imageUrlPath
      ? path.join(UPLOADS_DIR, path.basename(imageUrlPath))
      : file.path; // Use temp file path for local storage users
    // Minimal logging for speed

    const respondWithDuplicate = (duplicateDetection) => {
      scheduleUploadCleanup(imageUrlPath);
      return res.json({
        success: true,
        duplicate: true,
        sightingId: duplicateDetection._id,
        imageUrl: imageUrlPath,
        predictedSpecies: predicted_species,
        confidence: confidence,
        originalDetection: {
          id: duplicateDetection._id,
          species: duplicateDetection.analysis.predictedSpecies,
          confidence: duplicateDetection.analysis.confidence,
          detectedAt: duplicateDetection.createdAt,
          daysAgo: Math.floor(
            (Date.now() - duplicateDetection.createdAt) / (1000 * 60 * 60 * 24)
          ),
        },
      });
    };

    // 1) ML - Ultra-fast classification with 500ms timeout
    const options = {
      mode: "text",
      scriptPath: "../python/",
      args: [fullImagePath],
      // Unbuffered, optimized bytecode, ignore Python warnings on stdout
      pythonOptions: ["-u", "-O", "-W", "ignore"],
    };

    let predicted_species = "Unknown species";
    let confidence = 0.0;

    try {
      console.log("Running ML classification with path:", fullImagePath);
      const results = await PythonShell.run("ml_model.py", options);
      console.log("ML raw results:", results);

      if (results && results.length > 0) {
        try {
          const cleanedLines = results
            .map((line) => (typeof line === "string" ? line.trim() : ""))
            .filter(Boolean);

          const jsonLine = cleanedLines
            .slice()
            .reverse()
            .find((line) => line.startsWith("{") && line.endsWith("}"));

          let parsedPayload = null;

          if (jsonLine) {
            parsedPayload = JSON.parse(jsonLine);
          } else {
            const joined = cleanedLines.join("\n");
            const firstBrace = joined.indexOf("{");
            const lastBrace = joined.lastIndexOf("}");
            if (
              firstBrace !== -1 &&
              lastBrace !== -1 &&
              lastBrace > firstBrace
            ) {
              parsedPayload = JSON.parse(
                joined.slice(firstBrace, lastBrace + 1)
              );
            }
          }

          if (!parsedPayload) {
            throw new Error("No JSON payload found in ML output");
          }

          const mlResult = parsedPayload;
          console.log("ML parsed result:", mlResult);
          if (mlResult && mlResult.processing_time) {
            console.log("ML processing_time:", mlResult.processing_time);
          }

          if (!mlResult.error) {
            predicted_species = mlResult.predicted_species || "Unknown species";
            confidence = parseFloat(mlResult.confidence) || 0.0;
            console.log("ML classification successful:", {
              predicted_species,
              confidence,
            });

            // Do not reject low-confidence classifications (threshold = 0)
          } else {
            console.error("ML returned error:", mlResult.error);
          }
        } catch (parseError) {
          console.error(
            "Failed to parse ML result:",
            parseError,
            "Raw result:",
            results[0]
          );
          // Use defaults
        }
      } else {
        console.error("No results returned from ML classification");
      }
    } catch (mlError) {
      console.error("ML classification failed:", mlError);
      // Use defaults - don't fail the request
    }

    console.log(
      `[DEBUG] After ML classification: species="${predicted_species}", confidence=${confidence}`
    );

    // 2) Check for duplicate detection (for all detections with location)
    console.log(
      `[DUPLICATE] Pre-check: lat=${lat}, lng=${lng}, species="${predicted_species}"`
    );
    if (lat && lng && predicted_species !== "Unknown species") {
      console.log(
        `Checking for duplicates: species="${predicted_species}", confidence=${confidence}, lat=${lat}, lng=${lng}`
      );
      const duplicateDetection = await checkForDuplicate(
        predicted_species,
        confidence,
        parseFloat(lat),
        parseFloat(lng),
        req.auth.userId
      );

      if (duplicateDetection) {
        console.log(
          `Duplicate detection found: ${predicted_species} within ${DUPLICATE_CONFIG.RADIUS_METERS}m and ${DUPLICATE_CONFIG.TIME_WINDOW_DAYS} days`
        );

        // Return duplicate response without creating new database entry
        return respondWithDuplicate(duplicateDetection);
      }
    } else {
      console.log(
        `[DUPLICATE] Skipping duplicate check: lat=${!!lat}, lng=${!!lng}, species="${predicted_species}"`
      );
    }

    // 3) Validate and prepare location data
    let location = null;
    const parsedLat = parseFloat(lat);
    const parsedLng = parseFloat(lng);

    // Only set location if we have valid coordinates
    if (
      !isNaN(parsedLat) &&
      !isNaN(parsedLng) &&
      parsedLat !== 0 &&
      parsedLng !== 0 &&
      Math.abs(parsedLat) <= 90 &&
      Math.abs(parsedLng) <= 180
    ) {
      location = { type: "Point", coordinates: [parsedLng, parsedLat] };
    } else {
      console.warn(
        "Invalid coordinates provided, saving sighting without location:",
        { lat, lng, parsedLat, parsedLng }
      );
    }

    // 4) Sighting doc (LLM pending)
    const doc = await Sighting.create({
      owner: req.auth.userId,
      imagePath: imageUrlPath, // Store URL path in DB (null for local storage)
      localFileId: storageType === "local" ? localFileId : null, // Store local file ID for local storage
      storageType: storagePreference, // Track storage type
      videoSessionId: normalizedVideoSessionId,
      videoTimestamp: normalizedVideoTimestamp,
      location, // Will be null if coordinates are invalid
      analysis: {
        predictedSpecies: predicted_species,
        confidence,
        llm: { status: "pending" },
      },
    });

    // Post-insert duplicate safeguard (handles concurrent requests)
    if (lat && lng && predicted_species !== "Unknown species") {
      const raceDuplicate = await checkForDuplicate(
        predicted_species,
        confidence,
        parseFloat(lat),
        parseFloat(lng),
        req.auth.userId,
        { excludeId: doc._id }
      );

      if (raceDuplicate) {
        console.warn(
          `[DUPLICATE] Race condition detected for new sighting ${doc._id}, linking back to ${raceDuplicate._id}`
        );
        await Sighting.deleteOne({ _id: doc._id });
        return respondWithDuplicate(raceDuplicate);
      }
    }

    // Respond fast
    res.json({
      success: true,
      sightingId: doc._id,
      imageUrl: imageUrlPath, // Return URL path for frontend (null = use local storage)
      storageType: storagePreference,
      predictedSpecies: predicted_species,
      confidence,
    });

    // Immediately publish new sighting for real-time map updates
    publish(req.auth.userId, {
      type: "new_sighting",
      sighting: {
        _id: doc._id,
        analysis: doc.analysis,
        location: doc.location,
        createdAt: doc.createdAt,
        imagePath: doc.imagePath,
      },
    });

    // 3) Queue LLM analysis for parallel processing - analyze everything
    setImmediate(() => {
      // Capture userId in closure before async callback
      const userId = req.auth.userId;
      const sightingId = doc._id;

      const llmCallback = async (llm) => {
        try {
          // Batch database update and SSE publish for efficiency
          const [updateResult] = await Promise.allSettled([
            Sighting.updateOne(
              { _id: sightingId },
              {
                $set: {
                  "analysis.llm": {
                    summary: llm.summary || "",
                    details: llm.details,
                    status: "completed",
                  },
                },
              }
            ),
            Promise.resolve().then(() => {
              publish(userId, {
                type: "analysis_done",
                sightingId: sightingId,
                llm,
              });
              console.log(
                "SSE analysis_done event published for:",
                sightingId,
                "to user:",
                userId
              );
            }),
          ]);

          if (updateResult.status === "rejected") {
            console.error("Database update failed:", updateResult.reason);
          }

          // Video session status is already set after ML classification
          // No need to update again after LLM
        } catch (error) {
          console.error("LLM callback execution failed:", error);
          await Sighting.updateOne(
            { _id: sightingId },
            { $set: { "analysis.llm.status": "failed" } }
          );
        }
      };

      queueLLMAnalysis(
        sightingId,
        predicted_species,
        confidence,
        llmCallback
      ).catch(async (e) => {
        console.error("LLM queue processing failed:", e);
        await Sighting.updateOne(
          { _id: sightingId },
          { $set: { "analysis.llm.status": "failed" } }
        );
      });

      // Clean up temp file for local storage users
      if (storagePreference === "local" && file.path !== fullImagePath) {
        try {
          fs.unlinkSync(file.path);
          console.log(
            "Cleaned up temp file for local storage user:",
            file.path
          );
        } catch (cleanupError) {
          console.warn("Could not clean up temp file:", cleanupError);
        }
      }
    });
  } catch (e) {
    next(e);
  }
}
