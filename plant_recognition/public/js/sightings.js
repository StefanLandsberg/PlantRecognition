import { SightingsAPI } from "./api.js";

function fmtDate(s) {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s || "";
  }
}

function fmtLatLng(coords) {
  if (!Array.isArray(coords) || coords.length < 2) return "";
  const [lng, lat] = coords;
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
}

function pct(n) {
  if (n == null || isNaN(n)) return "";
  return (Number(n) * 100).toFixed(1) + "%";
}

async function load() {
  const tbody = document.querySelector("#sightings-table tbody");
  const empty = document.getElementById("sightings-empty");
  tbody.innerHTML = "";
  try {
    const { data } = await SightingsAPI.list("");
    if (!data || data.length === 0) {
      empty.style.display = "block";
      return;
    }
    empty.style.display = "none";
    for (const d of data) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="padding:.6rem;border-bottom:1px solid #24304a;">${fmtDate(
          d.capturedAt || d.createdAt
        )}</td>
        <td style="padding:.6rem;border-bottom:1px solid #24304a;">${
          d.analysis?.predictedSpecies || "Unknown"
        }</td>
        <td style="padding:.6rem;border-bottom:1px solid #24304a;">${pct(
          d.analysis?.confidence
        )}</td>
        <td style="padding:.6rem;border-bottom:1px solid #24304a;">${fmtLatLng(
          d.location?.coordinates
        )}</td>
        <td style="padding:.6rem;border-bottom:1px solid #24304a;">${
          d.fromVideo ? "Live video" : "Upload"
        }</td>
        <td style="padding:.6rem;border-bottom:1px solid #24304a;">
          ${
            d.imagePath
              ? `<a href="${d.imagePath}" target="_blank" rel="noopener">View</a>`
              : ""
          }
        </td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    console.error("Failed to load sightings", e);
    empty.textContent = "Failed to load sightings.";
    empty.style.display = "block";
  }
}

load();
