"use strict";
const byId = id => document.getElementById(id);
const form = byId("inspection-form"), status = byId("status"), result = byId("result");
let originalURL = null, active = false, generation = 0;
const modelModes = new Map();
function reset() {
  generation += 1;
  result.hidden = true;
  byId("decision").textContent = "";
  byId("heatmap").getContext("2d").clearRect(0, 0, byId("heatmap").width, byId("heatmap").height);
  if (originalURL) URL.revokeObjectURL(originalURL);
  originalURL = null;
  byId("original").removeAttribute("src");
}
byId("image").addEventListener("change", reset);
byId("model").addEventListener("change", reset);
async function initialize() {
  try {
    const [readyResponse, modelsResponse] = await Promise.all([fetch("/api/v1/ready"), fetch("/api/v1/models")]);
    const ready = await readyResponse.json(), registry = await modelsResponse.json();
    if (!modelsResponse.ok || !Array.isArray(registry.models)) throw new Error("Registry unavailable");
    for (const model of registry.models) {
      modelModes.set(model.model_id, model.mode);
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = `${model.model_id} — ${model.mode}`;
      byId("model").append(option);
    }
    byId("inspect").disabled = !ready.ready;
    status.textContent = ready.ready ? "Ready. Choose a model and image." : "Not ready. No authorized model or worker unavailable.";
  } catch (_) { status.textContent = "Service unavailable. No inspection decision."; }
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  if (active) return;
  reset();
  const current = generation;
  const file = byId("image").files[0], model = byId("model").value;
  if (!file || !model) { status.textContent = "Choose both a model and image."; return; }
  if (modelModes.get(model) === "native-development" && !byId("input-role").checked) { status.textContent = "Confirm authorized non-held-out input. No decision."; return; }
  if (file.size > 10 * 1024 * 1024) { status.textContent = "Image exceeds 10 MiB. No decision."; return; }
  active = true; byId("inspect").disabled = true;
  status.textContent = "Inspecting… previous decision cleared.";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const headers = {"Content-Type": file.type, "X-VisionGuard-Client": "inspection-v1"};
    if (byId("input-role").checked) headers["X-VisionGuard-Input-Role"] = "generated-or-development-non-held-out";
    const response = await fetch(`/api/v1/inspect/${encodeURIComponent(model)}`, {method: "POST", signal: controller.signal, headers, body: file});
    const data = await response.json();
    if (!response.ok || data.status !== "succeeded" || !["NORMAL", "ANOMALOUS"].includes(data.decision)) throw new Error(data.error?.message || "Invalid inference response");
    if (current !== generation) return;
    const map = new Image();
    map.src = `data:image/png;base64,${data.heatmap_png_base64}`;
    await map.decode();
    if (current !== generation) return;
    if (map.naturalWidth !== data.width || map.naturalHeight !== data.height) throw new Error("Heatmap dimensions invalid");
    const canvas = byId("heatmap");
    canvas.width = data.width; canvas.height = data.height;
    canvas.getContext("2d").drawImage(map, 0, 0);
    originalURL = URL.createObjectURL(file); byId("original").src = originalURL;
    byId("decision").textContent = data.decision;
    byId("identity").textContent = `${data.model.model_id} · artifact ${data.model.artifact_sha256} · preprocessing ${data.model.preprocessing_sha256}`;
    byId("score").textContent = `Score ${data.score} · threshold ${data.threshold} · strict score > threshold`;
    byId("mode").textContent = data.model.mode === "manufactured" ? "MANUFACTURED BACKEND — NOT NATIVE INFERENCE" : "Authorized development-state inference";
    byId("visualization").textContent = `Display only: min ${data.visualization.minimum}, max ${data.visualization.maximum}. ${data.visualization.constant_map ? "Constant map has no contrast." : "Independent of classification threshold."}`;
    result.hidden = false; status.textContent = "Inspection succeeded.";
  } catch (error) {
    reset(); status.textContent = `Inspection failed. No decision. ${error.message}`;
  } finally { clearTimeout(timer); active = false; byId("inspect").disabled = false; }
});
initialize();
