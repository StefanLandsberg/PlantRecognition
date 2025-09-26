export const manifestSchema = {
  observation_id: "string",
  species: "string",
  image_url: "string",
  lat: "float (-35 to -22)",
  lng: "float (16 to 33)",
  observed_on: "ISO8601 datetime",
  is_invasive: "boolean",
  source: "string"
};

const LAT_MIN = -35, LAT_MAX = -22, LNG_MIN = 16, LNG_MAX = 33;

export function validateRecord(record) {
  const errors = [];
  if (!record.observation_id) errors.push("observation_id missing");
  if (!record.species) errors.push("species missing");
  if (!record.image_url) errors.push("image_url missing");
  if (record.lat == null || isNaN(record.lat)) errors.push("lat missing/NaN");
  else if (record.lat < LAT_MIN || record.lat > LAT_MAX)
    errors.push(`lat out of range ${LAT_MIN}..${LAT_MAX}`);
  if (record.lng == null || isNaN(record.lng)) errors.push("lng missing/NaN");
  else if (record.lng < LNG_MIN || record.lng > LNG_MAX)
    errors.push(`lng out of range ${LNG_MIN}..${LNG_MAX}`);
  if (!record.observed_on || isNaN(new Date(record.observed_on)))
    errors.push("observed_on invalid");
  if (typeof record.is_invasive !== "boolean")
    errors.push("is_invasive must be boolean");
  return { valid: errors.length === 0, errors };
}
