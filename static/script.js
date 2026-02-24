(() => {
  const EMOJI = {
    Airplane: "✈️",
    Car: "🚗",
    Bird: "🐦",
    Cat: "🐱",
    Deer: "🦌",
    Dog: "🐶",
    Frog: "🐸",
    Horse: "🐴",
    Ship: "🚢",
    Truck: "🚛",
  };

  const els = {
    fileInput: document.getElementById("fileInput"),
    dropZone: document.getElementById("dropZone"),
    previewImg: document.getElementById("previewImg"),
    previewEmpty: document.getElementById("previewEmpty"),
    clearBtn: document.getElementById("clearBtn"),
    classifyBtn: document.getElementById("classifyBtn"),
    errorBox: document.getElementById("errorBox"),
    resultEmpty: document.getElementById("resultEmpty"),
    resultWrap: document.getElementById("resultWrap"),
    topEmoji: document.getElementById("topEmoji"),
    topLabel: document.getElementById("topLabel"),
    topConf: document.getElementById("topConf"),
    top3List: document.getElementById("top3List"),
  };

  if (
    !els.fileInput ||
    !els.dropZone ||
    !els.previewImg ||
    !els.previewEmpty ||
    !els.clearBtn ||
    !els.classifyBtn ||
    !els.errorBox ||
    !els.resultEmpty ||
    !els.resultWrap ||
    !els.topEmoji ||
    !els.topLabel ||
    !els.topConf ||
    !els.top3List
  ) {
    return;
  }

  let selectedFile = null;
  let isLoading = false;

  const setError = (msg) => {
    if (!msg) {
      els.errorBox.hidden = true;
      els.errorBox.textContent = "";
      return;
    }
    els.errorBox.hidden = false;
    els.errorBox.textContent = msg;
  };

  const setLoading = (loading) => {
    isLoading = loading;
    if (loading) {
      els.classifyBtn.classList.add("isLoading");
      els.classifyBtn.disabled = true;
      els.clearBtn.disabled = true;
      els.dropZone.setAttribute("aria-disabled", "true");
    } else {
      els.classifyBtn.classList.remove("isLoading");
      els.classifyBtn.disabled = !selectedFile;
      els.clearBtn.disabled = !selectedFile;
      els.dropZone.removeAttribute("aria-disabled");
    }
  };

  const clearResults = () => {
    els.resultWrap.hidden = true;
    els.resultEmpty.hidden = false;
    els.top3List.innerHTML = "";
  };

  const clearSelection = () => {
    selectedFile = null;
    els.fileInput.value = "";
    els.previewImg.removeAttribute("src");
    els.previewImg.closest(".preview").classList.remove("hasImage");
    els.previewEmpty.hidden = false;
    els.clearBtn.disabled = true;
    els.classifyBtn.disabled = true;
    setError("");
    clearResults();
  };

  const isValidImageFile = (file) => {
    if (!file) return false;
    if (file.type && file.type.startsWith("image/")) return true;
    const name = (file.name || "").toLowerCase();
    return /\.(png|jpe?g|webp|gif|bmp)$/i.test(name);
  };

  const setFile = (file) => {
    setError("");
    if (!isValidImageFile(file)) {
      clearSelection();
      setError("Please upload an image file (PNG, JPG/JPEG, WEBP, GIF).");
      return;
    }

    selectedFile = file;
    els.clearBtn.disabled = false;
    els.classifyBtn.disabled = false;

    const url = URL.createObjectURL(file);
    els.previewImg.src = url;
    els.previewImg.onload = () => URL.revokeObjectURL(url);
    els.previewImg.closest(".preview").classList.add("hasImage");
    els.previewEmpty.hidden = true;

    clearResults();
  };

  const renderTop3 = (top3) => {
    const rows = (Array.isArray(top3) ? top3 : []).slice(0, 3);
    els.top3List.innerHTML = "";

    for (const item of rows) {
      const className = item?.class_name ?? item?.class ?? item?.label ?? "Unknown";
      const conf = Number(item?.confidence ?? 0);
      const pct = Math.max(0, Math.min(100, conf * 100));
      const emoji = EMOJI[className] || "✨";

      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `
        <div class="rowHead">
          <div class="rowLabel"><span aria-hidden="true">${emoji}</span><span>${className}</span></div>
          <div class="rowPct">${pct.toFixed(1)}%</div>
        </div>
        <div class="bar" aria-label="${className} confidence ${pct.toFixed(1)} percent">
          <div class="barFill" style="width:0%"></div>
        </div>
      `;
      els.top3List.appendChild(row);

      const fill = row.querySelector(".barFill");
      if (fill) {
        // Trigger width transition on next frame (so it animates from 0%)
        requestAnimationFrame(() => {
          fill.style.width = `${pct.toFixed(1)}%`;
        });
      }
    }
  };

  const showResult = (data) => {
    const prediction = data?.prediction ?? "Unknown";
    const confidence = Number(data?.confidence ?? 0);
    const pct = Math.max(0, Math.min(100, confidence * 100));

    els.topEmoji.textContent = EMOJI[prediction] || "✨";
    els.topLabel.textContent = prediction;
    els.topConf.textContent = `${pct.toFixed(1)}% confidence`;

    renderTop3(data?.top3);

    els.resultEmpty.hidden = true;
    els.resultWrap.hidden = false;
  };

  const classify = async () => {
    if (!selectedFile || isLoading) return;

    setError("");
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", selectedFile);

      const res = await fetch("/predict", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const msg = data?.error || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      showResult(data);
    } catch (err) {
      clearResults();
      setError(err?.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Click / keyboard open file picker
  els.dropZone.addEventListener("click", () => {
    if (isLoading) return;
    els.fileInput.click();
  });

  els.dropZone.addEventListener("keydown", (e) => {
    if (isLoading) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files?.[0];
    if (file) setFile(file);
  });

  // Drag & drop
  const setDragOver = (on) => {
    els.dropZone.classList.toggle("isDragOver", on);
  };

  ["dragenter", "dragover"].forEach((evt) => {
    els.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (isLoading) return;
      setDragOver(true);
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    els.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);
    });
  });

  els.dropZone.addEventListener("drop", (e) => {
    if (isLoading) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });

  els.clearBtn.addEventListener("click", clearSelection);
  els.classifyBtn.addEventListener("click", classify);

  clearSelection();
})();

