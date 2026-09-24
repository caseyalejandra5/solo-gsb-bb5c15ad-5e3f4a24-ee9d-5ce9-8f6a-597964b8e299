/* 抗裁切水印实验台前端逻辑（原生 JS，无框架） */

const $ = (s) => document.querySelector(s);
const api = async (url, opts = {}) => {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
};

const state = {
  jobId: null,
  strength: "medium",
  embedMode: "manual",    // manual | auto（嵌入面板）
  detectMode: "normal",    // normal | orient（独立检测）
  originalUrl: null,
  watermarkedUrl: null,
  targetUrl: null,
  markedUrl: null,
  detections: [],          // 当前任务页内累积的检测结果（可归档）
  report: null,            // 最近生成的归档报告
};

/* ---------------- 嵌入：手动 / 自动模式 ---------------- */

$("#embed-mode-seg").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-value]");
  if (!btn) return;
  state.embedMode = btn.dataset.value;
  document.querySelectorAll("#embed-mode-seg button").forEach((b) =>
    b.classList.toggle("on", b === btn)
  );
  const isAuto = state.embedMode === "auto";
  $("#manual-strength-field").classList.toggle("hidden", isAuto);
  $("#auto-strength-field").classList.toggle("hidden", !isAuto);
  $("#btn-embed").textContent = isAuto ? "自动选强度并嵌入" : "嵌入水印";
});

$("#auto-jpeg-q").addEventListener("input", (e) =>
  $("#auto-jpeg-val").textContent = e.target.value);
$("#auto-crop-f").addEventListener("input", (e) =>
  $("#auto-crop-val").textContent = `${e.target.value}%`);
$("#auto-psnr").addEventListener("input", (e) =>
  $("#auto-psnr-val").textContent = `${e.target.value} dB`);


$("#strength-seg").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-value]");
  if (!btn) return;
  state.strength = btn.dataset.value;
  document.querySelectorAll("#strength-seg button").forEach((b) =>
    b.classList.toggle("on", b === btn)
  );
});

$("#embed-text").addEventListener("input", (e) => {
  const bytes = new TextEncoder().encode(e.target.value).length;
  $("#text-counter").textContent =
    `${bytes} 字节` + (bytes > 45 ? "（超出 45 字节容量上限）" : "");
  $("#text-counter").style.color = bytes > 45 ? "var(--bad)" : "";
});

$("#embed-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  state.originalUrl = URL.createObjectURL(f);
  showImage("original", state.originalUrl);
  switchTab("original");
});

$("#btn-embed").addEventListener("click", async () => {
  const file = $("#embed-file").files[0];
  const text = $("#embed-text").value.trim();
  const status = $("#embed-status");
  if (!file) return fail(status, "请先选择一张图片。");
  if (!text) return fail(status, "请输入水印文本。");

  if (state.embedMode === "auto") {
    return runAutoEmbed(file, text, status);
  }

  const fd = new FormData();
  fd.append("image", file);
  fd.append("text", text);
  fd.append("strength", state.strength);

  busy(status, "正在执行分块 DCT 嵌入…");
  try {
    const r = await api("/api/embed", { method: "POST", body: fd });
    applyEmbeddedJob(r, "手动选档：" + strengthLabel(r.strength));
    $("#auto-result").classList.add("hidden");
    ok(status, "嵌入成功，请查看「含水印图」标签。");
  } catch (err) {
    fail(status, err.message);
  }
});

async function runAutoEmbed(file, text, status) {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("text", text);
  fd.append("jpeg_quality", $("#auto-jpeg-q").value);
  fd.append("crop_fraction", parseInt($("#auto-crop-f").value, 10) / 100);
  fd.append("min_psnr_db", $("#auto-psnr").value);

  busy(status, "正在按 低 → 中 → 高 逐档嵌入并执行「先裁切后压缩 + 盲检」…");
  let r;
  try {
    r = await api("/api/embed-auto", { method: "POST", body: fd });
  } catch (err) {
    $("#auto-result").classList.add("hidden");
    return fail(status, err.message);
  }

  renderAutoTiers(r);
  if (!r.passed) {
    // 三档均未通过：服务端未创建任务
    $("#embed-meta").classList.add("hidden");
    $("#panel-attack").classList.add("disabled");
    $("#panel-matrix").classList.add("disabled");
    $("#panel-locate").classList.add("disabled");
    $("#btn-detect-attacked").disabled = true;
    fail(status, "低 / 中 / 高三档均未同时满足「恢复文本逐字一致」与「PSNR 达标」，未创建任务；请放宽攻击参数或降低 PSNR 下限后重试。");
    return;
  }

  applyEmbeddedJob(r, `自动选档：${strengthLabel(r.selected_strength)}（低 → 中 → 高首个通过档）`);
  ok(status, `自动选档完成：选用「${strengthLabel(r.selected_strength)}」强度，已保存任务，可下载与继续攻击 / 检测。`);
}

function applyEmbeddedJob(r, strengthNote) {
  state.jobId = r.job_id;
  state.watermarkedUrl = `/api/jobs/${r.job_id}/watermarked?t=${Date.now()}`;
  state.targetUrl = null;
  state.markedUrl = null;
  state.detections = [];
  state.report = null;
  renderDetectList();
  resetReportBox();
  resetMatrixPanel();
  resetLocatePanel();
  showImage("watermarked", state.watermarkedUrl);
  $("#img-target").removeAttribute("src");
  $("#img-marked").removeAttribute("src");

  $("#m-strength").textContent = strengthNote;
  $("#m-psnr").textContent = `${r.psnr_db} dB`;
  $("#m-tiles").textContent = `${r.full_tiles} 个`;
  $("#m-votes").textContent = `≈ ${r.votes_per_bit} 票/比特`;
  $("#m-cap").textContent = `${r.payload_bytes} / ${r.capacity_bytes} 字节`;
  $("#m-size").textContent = `${r.width} × ${r.height}`;
  $("#embed-meta").classList.remove("hidden");

  $("#panel-attack").classList.remove("disabled");
  $("#panel-matrix").classList.remove("disabled");
  $("#panel-locate").classList.remove("disabled");
  $("#btn-detect-attacked").disabled = false;
  document.querySelector('.step[data-step="2"]').classList.add("active");
  switchTab("watermarked");
}

const STRENGTH_TEXT = { low: "低", medium: "中", high: "高" };
function strengthLabel(s) {
  return STRENGTH_TEXT[s] ? `${STRENGTH_TEXT[s]}（${s}）` : String(s);
}

function renderAutoTiers(r) {
  const box = $("#auto-result");
  box.classList.remove("hidden");
  const req = r.request || {};
  const attackDesc =
    `JPEG Q${req.jpeg_quality} · 中心裁切 ${Math.round((req.crop_fraction ?? 0) * 100)}%` +
    ` · 最低 PSNR ${req.min_psnr_db} dB`;

  const summary = $("#auto-summary");
  if (r.passed) {
    summary.className = "auto-summary ok";
    summary.innerHTML =
      `攻击模拟 <b>${escapeHtml(attackDesc)}</b> · ` +
      `选用档位 <b class="auto-pick">${escapeHtml(strengthLabel(r.selected_strength))}</b>` +
      `（该档 PSNR ${r.selected_tier.psnr_db} dB，恢复文本与原文逐字一致）`;
  } else {
    summary.className = "auto-summary fail";
    summary.innerHTML =
      `攻击模拟 <b>${escapeHtml(attackDesc)}</b> · ` +
      `<b>三档均未通过，未创建任务</b>（逐档实测如下）`;
  }

  const head =
    `<thead><tr>
       <th>档位</th><th>嵌入 PSNR</th><th>PSNR 达标</th>
       <th>盲检恢复</th><th>文本一致</th><th>置信度</th><th>同步峰 / 一致率</th>
     </tr></thead>`;
  const rows = r.tiers.map((t) => {
    const chosen = r.passed && t.strength === r.selected_strength;
    const m = t.metrics || {};
    const recover = t.detected
      ? `<code>${escapeHtml(t.text || "（空）")}</code>`
      : `<span class="muted" title="${escapeHtml(t.reason || "")}">未恢复</span>`;
    return `<tr class="${chosen ? "chosen" : ""}" title="${escapeHtml(autoTierTitle(t))}">
      <td><b>${escapeHtml(t.strength_label)}（${t.strength}）</b>${chosen ? ' <span class="auto-pick-badge">★ 选用</span>' : ""}</td>
      <td>${t.psnr_db} dB</td>
      <td>${t.psnr_ok ? "✔" : "✘"}</td>
      <td class="auto-text">${recover}</td>
      <td>${t.text_match ? "✔ 逐字一致" : "✘"}</td>
      <td>${t.detected ? `${m.confidence ?? "—"}%` : "—"}</td>
      <td>${t.detected
        ? `${m.sync_score ?? "—"} / ${((m.sync_agreement ?? 0) * 100).toFixed(1)}%`
        : "—"}</td>
    </tr>`;
  }).join("");
  $("#auto-table").innerHTML = head + `<tbody>${rows}</tbody>`;
}

function autoTierTitle(t) {
  const m = t.metrics || {};
  const lines = [
    `${t.strength_label}档（${t.strength}）`,
    `嵌入 PSNR：${t.psnr_db} dB（${t.psnr_ok ? "达标" : "低于下限"}）`,
  ];
  if (t.detected) {
    lines.push(`恢复文本：${t.text || "（空）"}`);
    lines.push(`置信度 ${m.confidence}% · 同步峰 ${m.sync_score} · 一致率 ${(m.sync_agreement * 100).toFixed(1)}%`);
    lines.push(`瓦片 ${m.tiles_y}×${m.tiles_x} · 同步块 ${m.sync_blocks} · 载荷块 ${m.payload_blocks} · 票/bit ${m.votes_per_bit}`);
  } else {
    lines.push(`未恢复：${t.reason || "—"}`);
  }
  lines.push(`攻击后尺寸 ${t.width}×${t.height}px（吸附保留 ${Math.round(t.snapped_fraction_w * 100)}%×${Math.round(t.snapped_fraction_h * 100)}%）`);
  return lines.join("\n");
}


$("#btn-download-wm").addEventListener("click", () => {
  if (state.jobId) window.open(`/api/jobs/${state.jobId}/watermarked`, "_blank");
});

/* ---------------- 攻击 ---------------- */

$("#jpeg-q").addEventListener("input", (e) => $("#jpeg-val").textContent = e.target.value);
$("#crop-f").addEventListener("input", (e) => $("#crop-val").textContent = `${e.target.value}%`);

$("#btn-attack").addEventListener("click", async () => {
  const status = $("#attack-status");
  if (!state.jobId) return fail(status, "请先完成嵌入。");
  const q = parseInt($("#jpeg-q").value, 10);
  const crop = parseInt($("#crop-f").value, 10) / 100;
  busy(status, "正在施加 JPEG 重压缩 / 中心裁切…");
  try {
    const r = await api("/api/attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: state.jobId,
        jpeg_quality: q,
        crop_fraction: crop === 1 ? null : crop,
      }),
    });
    state.targetUrl = `/api/jobs/${state.jobId}/target?t=${Date.now()}`;
    showImage("target", state.targetUrl);
    switchTab("target");
    $("#attack-history").innerHTML = (r.attack_history || [])
      .map((h) => `<span class="chip">${
        h.type === "jpeg" ? `JPEG Q${h.quality}` : `中心裁切 ${Math.round(h.fraction * 100)}%`
      }</span>`)
      .join("") + `<span class="chip">${r.target_width}×${r.target_height}px</span>`;
    ok(status, "攻击已应用，可继续叠加攻击或直接检测。");
  } catch (err) {
    fail(status, err.message);
  }
});

$("#btn-attack-reset").addEventListener("click", async () => {
  const status = $("#attack-status");
  if (!state.jobId) return;
  busy(status, "重置中…");
  try {
    await api("/api/attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.jobId, jpeg_quality: 100, crop_fraction: 1 }),
    });
    state.targetUrl = `/api/jobs/${state.jobId}/target?t=${Date.now()}`;
    showImage("target", state.targetUrl);
    $("#attack-history").innerHTML = `<span class="chip">已重置（JPEG Q100、无裁切）</span>`;
    ok(status, "已重置为含水印图。");
  } catch (err) {
    fail(status, err.message);
  }
});

$("#btn-detect-attacked").addEventListener("click", () =>
  runDetect(null, state.jobId, "attack"));

/* ---------------- 鲁棒性矩阵评测（只读批量） ---------------- */

function readMatrixInputs(cls, kind) {
  const vals = [...document.querySelectorAll(`.${cls}`)]
    .map((el) => el.value.trim())
    .filter((v) => v !== "")
    .map((v) => Number(v));
  if (!vals.length) {
    throw new Error(kind === "q"
      ? "请至少填写 1 个 JPEG 质量。"
      : "请至少填写 1 个裁切保留比例。");
  }
  if (vals.some((v) => !Number.isFinite(v))) {
    throw new Error(kind === "q"
      ? "JPEG 质量必须是数字。"
      : "裁切保留比例必须是数字。");
  }
  if (new Set(vals).size !== vals.length) {
    throw new Error(kind === "q"
      ? "JPEG 质量不能重复。"
      : "裁切保留比例不能重复。");
  }
  if (kind === "q") {
    if (vals.some((v) => v < 20 || v > 100)) {
      throw new Error("JPEG 质量需在 20–100 之间。");
    }
  } else if (vals.some((v) => v < 30 || v > 100)) {
    throw new Error("裁切保留比例需在 30%–100% 之间。");
  }
  return vals;
}

$("#btn-matrix").addEventListener("click", async () => {
  const status = $("#matrix-status");
  if (!state.jobId) return fail(status, "请先完成嵌入。");
  let qualities, cropPercents;
  try {
    qualities = readMatrixInputs("matrix-q", "q");
    cropPercents = readMatrixInputs("matrix-c", "c");
  } catch (err) {
    return fail(status, err.message);
  }
  busy(status, `正在对 ${qualities.length * cropPercents.length} 个组合逐一执行先裁切后压缩与检测…`);
  try {
    const r = await api(
      `/api/jobs/${state.jobId}/robustness-matrix`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jpeg_qualities: qualities,
          crop_fractions: cropPercents.map((p) => p / 100),
        }),
      }
    );
    renderMatrix(r);
    ok(status, `矩阵评测完成：${r.counts.successful}/${r.counts.cells} 格恢复一致，最严成功边界 ${r.counts.boundary} 格（未改写任务数据）。`);
  } catch (err) {
    $("#matrix-result").classList.add("hidden");
    fail(status, err.message);
  }
});

function cellClass(c) {
  if (!c.ok) return "fail";
  if (c.text_match) return "ok";
  if (c.detected) return "diff";
  return "fail";
}

function cellTitle(c) {
  const lines = [
    `JPEG Q${c.jpeg_quality} × 裁切 ${Math.round(c.crop_fraction * 100)}%`,
  ];
  if (c.ok) {
    if (c.detected) {
      lines.push(`恢复文本：${c.text || "（空）"}`);
    } else {
      lines.push(`未恢复：${c.reason || "—"}`);
    }
    if (c.metrics) {
      const m = c.metrics;
      lines.push(
        `置信度 ${m.confidence}% · 同步峰 ${m.sync_score} · 一致率 ${(m.sync_agreement * 100).toFixed(1)}%`
      );
      lines.push(
        `瓦片 ${m.tiles_y}×${m.tiles_x} · 同步块 ${m.sync_blocks} · 载荷块 ${m.payload_blocks} · 票/bit ${m.votes_per_bit}`
      );
      lines.push(
        `失真块 ${m.marked_blocks}/${m.total_blocks}（${(m.marked_fraction * 100).toFixed(1)}%）`
      );
    }
    if (c.width != null) {
      lines.push(`攻击后尺寸 ${c.width}×${c.height}px（吸附后 ${Math.round((c.snapped_fraction_w ?? 0) * 100)}%×${Math.round((c.snapped_fraction_h ?? 0) * 100)}%）`);
    }
  } else {
    lines.push(c.reason || "评测执行失败");
  }
  return lines.join("\n");
}

function renderMatrix(r) {
  $("#matrix-result").classList.remove("hidden");

  // 显示顺序：质量从高到轻（列，左→右更轻），保留比例从大到小（行，上→下更轻）
  const qs = [...r.jpeg_qualities].sort((a, b) => b - a);
  const fs = [...r.crop_fractions].sort((a, b) => b - a);
  const key = (q, f) => `${q}:${f}`;
  const map = new Map(r.cells.map((c) => [key(c.jpeg_quality, c.crop_fraction), c]));

  const tbl = $("#matrix-table");
  tbl.innerHTML = "";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.innerHTML = `<th title="行=中心裁切保留比例，列=JPEG 质量">裁切 ＼ JPEG</th>` +
    qs.map((q) => `<th>Q${q}</th>`).join("");
  thead.appendChild(headRow);
  tbl.appendChild(thead);

  const tbody = document.createElement("tbody");
  fs.forEach((f) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = `${Math.round(f * 100)}%`;
    tr.appendChild(th);
    qs.forEach((q) => {
      const c = map.get(key(q, f));
      const td = document.createElement("td");
      if (!c) {
        td.className = "fail";
        td.textContent = "—";
      } else {
        td.className = `mx-cell ${cellClass(c)}` + (c.on_boundary ? " boundary" : "");
        const mark = c.text_match
          ? (c.on_boundary ? "★" : "✔")
          : c.detected ? "≠" : "✘";
        const conf = c.ok && c.metrics ? `${c.metrics.confidence}%` : "—";
        td.innerHTML = `<div class="mx-mark">${mark}</div><div class="mx-conf">${conf}</div>`;
        td.title = cellTitle(c);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);

  $("#matrix-summary").innerHTML =
    `任务原文 <code>${escapeHtml(r.text || "（空）")}</code> · 共 ${r.counts.cells} 格 · ` +
    `<b class="mx-ok-text">一致 ${r.counts.successful}</b> · ` +
    `<span class="mx-fail-text">未恢复/不一致 ${r.counts.failed}</span> · ` +
    `最严成功边界 <b>${r.counts.boundary}</b> 格`;

  const box = $("#matrix-boundary-list");
  box.innerHTML = "";
  if (r.boundary.length) {
    box.innerHTML =
      `<span class="muted">二维最严成功边界（只含实测格，不做相邻推断）：</span>` +
      r.boundary.map((b) =>
        `<span class="chip boundary-chip">★ Q${b.jpeg_quality} · 裁切 ${Math.round(b.crop_fraction * 100)}%</span>`
      ).join(" ");
  } else {
    box.innerHTML = `<span class="mx-fail-text">所选组合中没有任何一格恢复出与原文一致的文本，因此不存在成功边界。</span>`;
  }
}

/* ---------------- 传播副本来源定位（只读，不写任务/归档） ---------------- */

$("#btn-locate").addEventListener("click", async () => {
  const status = $("#locate-status");
  const file = $("#locate-file").files[0];
  if (!state.jobId) return fail(status, "请先完成嵌入。");
  if (!file) return fail(status, "请先选择一张传播副本图片。");

  const fd = new FormData();
  fd.append("image", file);
  busy(status, "正在提取局部特征、执行 KNN 匹配与 RANSAC 几何校验…");
  try {
    const r = await api(`/api/jobs/${state.jobId}/locate`,
                        { method: "POST", body: fd });
    renderLocateResult(r);
    if (r.located) {
      ok(status, `定位成功：副本覆盖参照图 ${r.coverage_percent}%（内点 ${r.inlier_count}/${r.good_matches}）。结果只读，未写任务文件与检测归档。`);
    } else {
      fail(status, "无法定位：" + r.reason);
    }
  } catch (err) {
    $("#locate-result").classList.add("hidden");
    fail(status, err.message);
  }
});

const LOCATE_CHECK_LABELS = {
  good_matches: "良好匹配数",
  homography: "RANSAC 单应估计",
  inlier_count: "内点数量",
  inlier_ratio: "内点比例",
  quad_finite: "四角投影有效",
  quad_shape: "四边形形状（凸性/夹角/尺度）",
  quad_bounds: "四边形边界",
  coverage: "面积覆盖率",
  reprojection_rms: "重投影误差",
};

function renderLocateResult(r) {
  const box = $("#locate-result");
  box.classList.remove("hidden");
  const verdict = $("#locate-verdict");
  const failBox = $("#locate-fail-reason");
  const img = $("#locate-preview");
  const metrics = $("#locate-metrics");
  const corners = $("#locate-corners");

  if (r.located) {
    verdict.className = "locate-verdict ok";
    verdict.textContent =
      `◎ 定位成功：该副本来自本任务含水印图，覆盖率 ${r.coverage_percent}%`;
    failBox.classList.add("hidden");
    failBox.textContent = "";
    img.src = r.preview_data_url;
    img.classList.remove("hidden");

    metrics.innerHTML =
      locateRow("参照图尺寸", `${r.reference_size.width} × ${r.reference_size.height}px`) +
      locateRow("副本尺寸", `${r.query_size.width} × ${r.query_size.height}px`) +
      locateRow("良好匹配 / 内点", `${r.good_matches} / ${r.inlier_count}（${(r.inlier_ratio * 100).toFixed(1)}%）`) +
      locateRow("重投影 RMS", `${r.rms_px} px（参照图口径）`) +
      locateRow("四边形面积", `${r.quad_area_px2.toLocaleString()} px²`) +
      locateRow("覆盖率", `${r.coverage_percent}%`) +
      locateRow("最小内角", `${r.min_angle_deg}°`) +
      locateRow("两轴尺度比", `${r.axis_scale_ratio} : 1`);

    corners.innerHTML =
      `<span class="muted">副本四角在参照图中的坐标（左上原点，像素）：</span>` +
      r.quad.map((c) =>
        `<span class="chip locate-corner">${escapeHtml(c.label)}：` +
        `(${c.x}, ${c.y})</span>`).join(" ");
  } else {
    verdict.className = "locate-verdict fail";
    verdict.textContent = "✘ 无法定位（不提供伪位置）";
    failBox.classList.remove("hidden");
    failBox.innerHTML =
      `<b>失败判定项：${escapeHtml(LOCATE_CHECK_LABELS[r.failed_check] || r.failed_check)}</b>` +
      `<div>${escapeHtml(r.reason || "未知原因")}</div>`;
    img.classList.add("hidden");
    img.removeAttribute("src");
    metrics.innerHTML =
      locateRow("参照图特征点", r.keypoints_ref ?? "—") +
      locateRow("副本特征点", r.keypoints_query ?? "—") +
      locateRow("良好匹配", r.good_matches ?? "—") +
      locateRow("RANSAC 内点",
        r.inlier_count == null ? "—"
          : `${r.inlier_count}（${((r.inlier_ratio ?? 0) * 100).toFixed(1)}%）`);
    corners.innerHTML =
      `<span class="muted">已通过 / 未通过的判定：</span>` +
      Object.entries(r.checks || {}).map(([k, v]) =>
        `<span class="chip locate-check ${v ? "pass" : "fail"}">` +
        `${v ? "✔" : "✘"} ${escapeHtml(LOCATE_CHECK_LABELS[k] || k)}</span>`
      ).join(" ");
  }
}

function locateRow(label, value) {
  return `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd>`;
}

function resetLocatePanel() {
  $("#panel-locate").classList.add("disabled");
  $("#locate-file").value = "";
  const st = $("#locate-status");
  st.className = "status";
  st.textContent = "";
  const box = $("#locate-result");
  box.classList.add("hidden");
  const img = $("#locate-preview");
  img.removeAttribute("src");
  img.classList.add("hidden");
}

/* ---------------- 独立检测 ---------------- */

$("#detect-mode-seg").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-value]");
  if (!btn) return;
  state.detectMode = btn.dataset.value;
  document.querySelectorAll("#detect-mode-seg button").forEach((b) =>
    b.classList.toggle("on", b === btn)
  );
});

$("#btn-detect-upload").addEventListener("click", async () => {
  const file = $("#detect-file").files[0];
  const status = $("#detect-upload-status");
  if (!file) return fail(status, "请先选择待检测图片。");
  await runDetect(file, null, "upload", status,
    state.detectMode === "orient" ? "1" : "0");
});

/* ---------------- 检测核心 ---------------- */

async function runDetect(file, jobId, source, statusEl, autoOrient = "0") {
  const status = statusEl || $("#attack-status");
  busy(status, autoOrient === "1"
    ? "正在 8 个方向上逐一执行同步搜索与多数表决…"
    : "正在执行 64 相移同步搜索与多数表决…");
  const fd = new FormData();
  if (file) fd.append("image", file);
  if (jobId) fd.append("job_id", jobId);
  fd.append("auto_orient", autoOrient);
  try {
    const r = await api("/api/detect", { method: "POST", body: fd });
    renderResult(r, source);
    if (r.orientation_search) {
      const os = r.orientation_search;
      if (!r.detected) {
        ok(status, `方向检索完成：${os.orientations_tried} 个方向均未通过载荷校验，未能恢复水印。`);
      } else if (os.conflict) {
        ok(status, `检测完成，但有效方向恢复出 ${os.conflict_texts.length} 种不同文本，存在冲突，请查看候选。`);
      } else {
        ok(status, `检测完成，水印方向「${os.chosen_label}」已恢复。`);
      }
    } else {
      ok(status, r.detected ? "检测完成，水印已恢复。" : "检测完成，但未能恢复水印。");
    }
  } catch (err) {
    fail(status, err.message);
  }
}

function renderResult(r, source) {
  const panel = $("#result-panel");
  panel.classList.remove("hidden");
  const m = r.metrics;

  const verdict = $("#detect-verdict");
  if (r.detected) {
    verdict.className = "verdict ok";
    verdict.textContent = `✔ 检测到有效水印（${m.confidence}% 置信度）`;
  } else {
    verdict.className = "verdict fail";
    verdict.textContent = `✘ 未检测到可校验的水印：${r.reason || "未知原因"}`;
  }

  $("#detect-text").textContent = r.detected ? r.text : "（未恢复）";

  setBar("overall", m.confidence, `${m.confidence}%`);
  const syncPct = Math.round(Math.min(1, Math.abs(m.sync_score)) * 100);
  setBar("sync", syncPct, m.sync_score.toFixed(3));
  setBar("agree", Math.round(m.sync_agreement * 100), `${(m.sync_agreement * 100).toFixed(1)}%`);

  const grid = $("#bit-grid");
  grid.innerHTML = "";
  (m.bit_confidences || []).forEach((c) => {
    const cell = document.createElement("i");
    const v = Math.round(c * 100);
    cell.style.background = v > 80
      ? "var(--accent-2)" : v > 50 ? "var(--warn)" : "var(--bad)";
    cell.title = `bit confidence ${v}%`;
    grid.appendChild(cell);
  });

  $("#r-tiles").textContent = `${m.tiles_y} × ${m.tiles_x}（共 ${m.tiles_y * m.tiles_x} 瓦）`;
  $("#r-phase").textContent = `(${m.phase[0]}, ${m.phase[1]}) 块`;
  $("#r-blocks").textContent = `${m.sync_blocks} / ${m.payload_blocks}`;
  $("#r-votes").textContent = m.votes_per_bit || "—";
  $("#r-marked").textContent =
    `${m.marked_blocks} / ${m.total_blocks} 块（${(m.marked_fraction * 100).toFixed(1)}%）`;

  renderOrientation(r.orientation_search, r.detected);

  if (r.marked_url) {
    state.markedUrl = `${r.marked_url}?t=${Date.now()}`;
    showImage("marked", state.markedUrl);
    document.querySelector('.step[data-step="3"]').classList.add("active");
    switchTab("marked");
  }

  recordDetection(r);
}

/* ---------------- 自动方向检索结果 ---------------- */

function renderOrientation(os, detected) {
  const box = $("#orientation-box");
  const rowDt = $("#r-orient-dt");
  const rowDd = $("#r-orient");
  if (!os) {
    box.classList.add("hidden");
    rowDt.style.display = "none";
    rowDd.style.display = "none";
    return;
  }
  box.classList.remove("hidden");
  rowDt.style.display = "";
  rowDd.style.display = "";

  const chosen = os.chosen_index != null
    ? os.candidates[os.chosen_index] : null;
  $("#orient-chosen").textContent = chosen
    ? `入选方向：${chosen.label} · ${chosen.confidence}%`
    : "无有效方向";
  $("#r-orient").textContent = chosen ? chosen.label : "—（未通过载荷校验）";
  $("#orient-valid").textContent =
    `${os.valid_candidate_count} / ${os.orientations_tried} 个方向通过载荷校验`;

  const conflict = $("#orient-conflict");
  if (os.conflict) {
    conflict.classList.remove("hidden");
    conflict.innerHTML =
      "⚠ 有效方向恢复出不同文本（冲突），最佳候选并非唯一结论：" +
      os.conflict_texts.map((t) => `<code>${escapeHtml(t)}</code>`).join("、");
  } else {
    conflict.classList.add("hidden");
    conflict.textContent = "";
  }

  const box2 = $("#orient-candidates");
  box2.innerHTML = "";
  os.candidates.forEach((c) => {
    const row = document.createElement("div");
    row.className = "orient-cand" + (c.detected ? " valid" : " invalid")
      + (c.index === os.chosen_index ? " chosen" : "");
    const mark = c.index === os.chosen_index ? "★ " : "";
    const body = c.detected
      ? `<b>${mark}${escapeHtml(c.label)}</b>
         <code>${escapeHtml(c.text)}</code>
         <span>${c.confidence}% · 同步 ${Math.abs(c.sync_score).toFixed(3)}</span>`
      : `<b>${mark}${escapeHtml(c.label)}</b>
         <span class="muted">未通过载荷校验 · 同步 ${Math.abs(c.sync_score).toFixed(3)}</span>`;
    row.innerHTML = body;
    row.title = c.detected ? "" : c.reason || "";
    box2.appendChild(row);
  });
}

/* ---------------- 检测报告归档 ---------------- */

const SOURCE_LABELS = {
  target: "受攻击图",
  watermarked: "含水印图",
  original: "原图",
  uploaded: "上传图",
};

function recordDetection(r) {
  // 跨任务（例如先嵌入后又用独立上传）时清空旧任务的检测记录
  if (!state.jobId || state.jobId !== r.job_id) {
    state.jobId = r.job_id;
    state.detections = [];
    resetReportBox();
  }
  state.detections.unshift({
    detection_id: r.detection_id,
    job_id: r.job_id,
    source: r.source,
    detected: r.detected,
    text: r.text || "",
    confidence: r.metrics.confidence,
    at: r.detected_at_iso || r.detected_at,
    marked_det_url: r.marked_det_url,
    orientation_label:
      r.orientation_search && r.orientation_search.chosen_index != null
        ? r.orientation_search.chosen_label
        : null,
    orientation_conflict:
      !!(r.orientation_search && r.orientation_search.conflict),
    checked: true,
  });
  renderDetectList();
}

function renderDetectList() {
  const box = $("#detect-list");
  box.innerHTML = "";
  if (!state.detections.length) {
    box.innerHTML = `<div class="muted">暂无检测结果，完成检测后可在此选择并归档。</div>`;
  } else {
    state.detections.forEach((d, i) => {
      const label = document.createElement("label");
      label.className = "detect-item";
      const time = String(d.at || "").replace("T", " ").replace("Z", " UTC");
      const orient = d.orientation_label
        ? ` · 方向：${escapeHtml(d.orientation_label)}${d.orientation_conflict ? " ⚠冲突" : ""}`
        : "";
      label.innerHTML = `
        <input type="checkbox" ${d.checked ? "checked" : ""} data-i="${i}" />
        <span class="det-badge ${d.detected ? "ok" : "fail"}">
          ${d.detected ? "✔" : "✘"}
        </span>
        <span class="det-main">
          <b>${escapeHtml(d.detected ? d.text : "未恢复水印")}</b>
          <small>${SOURCE_LABELS[d.source] || d.source} · 置信度 ${d.confidence ?? "—"}%${orient} · ${escapeHtml(time)}</small>
        </span>`;
      box.appendChild(label);
    });
  }
  updateArchiveButton();
}

$("#detect-list").addEventListener("change", (e) => {
  const cb = e.target.closest('input[type="checkbox"]');
  if (!cb) return;
  state.detections[+cb.dataset.i].checked = cb.checked;
  updateArchiveButton();
});

function selectedDetections() {
  return state.detections.filter((d) => d.checked);
}

function updateArchiveButton() {
  const n = selectedDetections().length;
  const btn = $("#btn-archive");
  btn.disabled = n === 0 || !state.jobId;
  btn.textContent = n
    ? `归档所选 ${n} 次检测为报告`
    : "归档所选检测为报告";
}

$("#btn-archive").addEventListener("click", async () => {
  const status = $("#archive-status");
  const picked = selectedDetections();
  if (!picked.length || !state.jobId) {
    return fail(status, "请至少勾选一次检测结果。");
  }
  busy(status, "正在生成规范化报告、计算 SHA-256 清单并写入归档…");
  resetReportBox();
  try {
    const r = await api(`/api/jobs/${state.jobId}/reports`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ detection_ids: picked.map((d) => d.detection_id) }),
    });
    state.report = {
      report_id: r.report_id,
      time: r.generated_at_iso,
      entries: r.entries,
      seq: r.archive && r.archive.seq,
    };
    renderReportBox();
    ok(status, `报告已生成并追加到归档记录（#${r.archive.seq}，清单 ${r.entries} 项）。`);
  } catch (err) {
    fail(status, err.message);
  }
});

function renderReportBox() {
  const rep = state.report;
  if (!rep) return;
  $("#report-box").classList.remove("hidden");
  $("#report-id").textContent = rep.report_id;
  $("#report-time").textContent =
    `${String(rep.time).replace("T", " ").replace("Z", " UTC")} · 清单 ${rep.entries} 项 · 归档 #${rep.seq}`;
  const v = $("#report-verify");
  v.className = "verify pending";
  v.textContent = "尚未校验";
  $("#report-verify-detail").textContent = "";
}

function resetReportBox() {
  state.report = null;
  $("#report-box").classList.add("hidden");
  const v = $("#report-verify");
  v.className = "verify pending";
  v.textContent = "尚未校验";
  $("#report-verify-detail").textContent = "";
}

$("#btn-download-report").addEventListener("click", () => {
  if (state.report) {
    window.open(`/api/reports/${state.report.report_id}?download=1`, "_blank");
  }
});

$("#btn-verify-report").addEventListener("click", async () => {
  if (!state.report) return;
  const v = $("#report-verify");
  const detail = $("#report-verify-detail");
  v.className = "verify busy";
  v.textContent = "正在重新计算清单摘要…";
  detail.textContent = "";
  try {
    const res = await fetch(`/api/reports/${state.report.report_id}/verify`,
                            { method: "POST" });
    const r = await res.json();
    if (r.valid && r.archive && r.archive.chain_ok) {
      v.className = "verify ok";
      v.textContent = `✔ 校验通过：${r.entries_total} 项清单全部一致，归档链完整（${r.archive.log_entries} 条记录）`;
      detail.textContent = "";
    } else {
      v.className = "verify fail";
      const parts = [];
      if (r.first_issue) {
        const it = r.first_issue;
        parts.push(`首个异常：${it.path}（${issueLabel(it.kind)}）`);
      }
      if (r.archive && !r.archive.chain_ok && r.archive.issue) {
        parts.push(`归档记录异常：${issueLabel(r.archive.issue.kind)}` +
          (r.archive.issue.seq ? ` #${r.archive.issue.seq}` : ""));
      }
      v.textContent = "✘ 校验未通过";
      detail.textContent = parts.join("；") || "清单或归档链存在不一致。";
    }
  } catch (err) {
    v.className = "verify fail";
    v.textContent = "✘ 校验请求失败";
    detail.textContent = err.message;
  }
});

function issueLabel(kind) {
  return {
    missing: "文件缺失",
    digest_mismatch: "SHA-256 摘要不一致",
    log_missing: "归档日志缺失",
    log_corrupt: "归档日志记录损坏",
    entry_digest_mismatch: "归档记录摘要不一致",
    chain_broken: "哈希链断裂（记录可能被删改）",
    report_not_in_archive: "报告未收录进归档",
    report_digest_mismatch: "归档摘要与当前报告不一致",
  }[kind] || kind;
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

$("#btn-download-marked").addEventListener("click", () => {
  if (state.markedUrl) window.open(state.markedUrl.split("?")[0], "_blank");
});

/* ---------------- 图片隐私检查（EXIF，只读不落盘） ---------------- */

$("#btn-privacy").addEventListener("click", async () => {
  const file = $("#privacy-file").files[0];
  const status = $("#privacy-status");
  if (!file) return fail(status, "请先选择一张 PNG / JPEG 图片。");
  if (!/^image\/(png|jpe?g)$/i.test(file.type)) {
    return fail(status, "仅支持 PNG / JPEG 图片。");
  }

  const fd = new FormData();
  fd.append("image", file);
  busy(status, "正在解析 EXIF 拍摄时间 / 相机 / GPS…");
  try {
    const r = await api("/api/privacy-check", { method: "POST", body: fd });
    renderPrivacyResult(r);
    ok(status, "检查完成（图片与元数据均未保存）。");
  } catch (err) {
    $("#privacy-result").classList.add("hidden");
    fail(status, err.message);
  }
});

function renderPrivacyResult(r) {
  $("#privacy-result").classList.remove("hidden");

  const warnBox = $("#privacy-warnings");
  warnBox.innerHTML = "";
  (r.warnings || []).forEach((w) => {
    const div = document.createElement("div");
    div.className = "privacy-warn";
    div.textContent = "⚠ " + w;
    warnBox.appendChild(div);
  });

  const time = r.date_time
    ? `${escapeHtml(r.date_time)}${r.date_time_iso ? ` <small>(${escapeHtml(r.date_time_iso)})</small>` : ""}`
    : "无";
  $("#pv-time").innerHTML = time;
  $("#pv-make").textContent = r.camera_make || "无";
  $("#pv-model").textContent = r.camera_model || "无";

  if (r.gps) {
    $("#pv-lat").innerHTML = gpsText(r.gps.latitude, r.gps.latitude_dms, r.gps.latitude_ref);
    $("#pv-lon").innerHTML = gpsText(r.gps.longitude, r.gps.longitude_dms, r.gps.longitude_ref);
  } else {
    $("#pv-lat").textContent = "无";
    $("#pv-lon").textContent = "无";
  }
  $("#pv-meta").textContent =
    `${r.format || "—"} · ${r.width} × ${r.height}px · ${formatBytes(r.bytes)}` +
    (r.exif_present ? "" : "（无 EXIF）");
}

function gpsText(decimal, dms, ref) {
  if (decimal === null || decimal === undefined) return "无";
  const dir = ref ? ` ${escapeHtml(ref)}` : "";
  const dmsHtml = dms ? `<small>${escapeHtml(dms)}${dir}</small>` : "";
  return `${decimal}${dmsHtml ? "<br>" + dmsHtml : ""}`;
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

/* ---------------- UI 辅助 ---------------- */

function resetMatrixPanel() {
  $("#panel-matrix").classList.add("disabled");
  $("#matrix-result").classList.add("hidden");
  const st = $("#matrix-status");
  st.className = "status";
  st.textContent = "";
}

function setBar(name, pct, label) {
  $("#bar-" + name).style.width = `${Math.max(0, Math.min(100, pct))}%`;
  $("#c-" + name).textContent = label;
}

function showImage(kind, url) {
  const img = $("#img-" + kind);
  img.src = url;
  const pane = img.closest(".preview-pane");
  pane.querySelector(".placeholder").style.display = "none";
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("on", t.dataset.tab === name));
  document.querySelectorAll(".preview-pane").forEach((p) =>
    p.classList.toggle("on", p.dataset.pane === name));
}

document.querySelectorAll(".preview-tabs .tab").forEach((t) =>
  t.addEventListener("click", () => switchTab(t.dataset.tab)));

function busy(el, msg) { el.className = "status busy"; el.textContent = msg; }
function ok(el, msg) { el.className = "status ok"; el.textContent = "✔ " + msg; }
function fail(el, msg) { el.className = "status error"; el.textContent = "✘ " + msg; }
