(() => {
  const API_BASE = "api/v1";
  const REQUEST_TIMEOUT_MS = 8000;
  const ROLE_OPTIONS = {
    buyer: {
      label: "买方",
      intentLabel: "预计月 Token 用量",
      intents: ["少于 1M", "1M – 10M", "10M – 100M", "100M 以上"],
    },
    supplier: {
      label: "节点方",
      intentLabel: "可提供的 GPU / 推理能力",
      intents: ["单张消费级 GPU", "多张消费级 GPU", "数据中心 GPU", "已有推理集群"],
    },
  };

  const toast = document.querySelector(".toast");
  let toastTimer = 0;

  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 4200);
  }

  async function requestJson(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(`${API_BASE}/${path}`, {
        ...options,
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `请求失败（${response.status}）`);
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("服务响应超时，请稍后再试");
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function setupServiceStatus() {
    const indicator = document.querySelector("[data-service-indicator]");
    const label = document.querySelector("[data-service-label]");
    const nodeCount = document.querySelector("[data-node-count]");
    const modelCount = document.querySelector("[data-model-count]");
    if (!indicator || !label || !nodeCount || !modelCount) return;

    requestJson("status")
      .then((status) => {
        const intakeOnline =
          status.status === "operational" && status.accepting_applications === true;
        indicator.classList.toggle("online", intakeOnline);
        indicator.classList.toggle("offline", !intakeOnline);
        indicator.classList.remove("checking");
        label.textContent = intakeOnline
          ? "EARLY ACCESS / INTAKE ONLINE"
          : "EARLY ACCESS / INTAKE PAUSED";
        nodeCount.textContent = `${Number(status.connected_nodes) || 0} connected`;
        modelCount.textContent = `${Number(status.advertised_models) || 0} advertised`;
      })
      .catch(() => {
        indicator.classList.add("offline");
        indicator.classList.remove("checking");
        label.textContent = "EARLY ACCESS / STATUS UNAVAILABLE";
        nodeCount.textContent = "status unavailable";
        modelCount.textContent = "status unavailable";
      });
  }

  function setupNetworkTilt() {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const network = document.querySelector(".network-shell");
    if (!network) return;

    let frame = 0;
    const reset = () => {
      network.style.setProperty("--tilt-x", "0deg");
      network.style.setProperty("--tilt-y", "0deg");
    };

    network.addEventListener(
      "pointermove",
      (event) => {
        if (reduceMotion.matches || event.pointerType === "touch") return;
        const rect = network.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
          network.style.setProperty("--tilt-x", `${(x - 0.5) * 3.2}deg`);
          network.style.setProperty("--tilt-y", `${(0.5 - y) * 3.2}deg`);
        });
      },
      { passive: true },
    );
    network.addEventListener("pointerleave", reset, { passive: true });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) reset();
    });
  }

  function setupCodeTabs() {
    const tabs = document.querySelectorAll("[data-code-tab]");
    const panels = document.querySelectorAll("[data-code-panel]");
    const fileName = document.querySelector(".terminal-bar strong");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const selected = tab.dataset.codeTab;
        tabs.forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
        panels.forEach((panel) => {
          panel.hidden = panel.dataset.codePanel !== selected;
        });
        if (fileName) fileName.textContent = "node-agent.sh";
      });
    });

    document.querySelector(".copy-button")?.addEventListener("click", async (event) => {
      const activeCode = document.querySelector("[data-code-panel]:not([hidden]) code");
      if (!activeCode) return;
      try {
        await navigator.clipboard.writeText(activeCode.textContent);
        event.currentTarget.textContent = "COPIED";
        showToast("代码已复制到剪贴板。");
        window.setTimeout(() => {
          event.currentTarget.textContent = "COPY";
        }, 1800);
      } catch {
        showToast("浏览器未允许自动复制，请手动选择代码。");
      }
    });
  }

  function setupRoleTabs() {
    const tabs = document.querySelectorAll("[data-role]");
    const panels = document.querySelectorAll("[data-role-panel]");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const selected = tab.dataset.role;
        tabs.forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
        panels.forEach((panel) => {
          panel.hidden = panel.dataset.rolePanel !== selected;
        });
      });
    });
  }

  function setupAccessDialog() {
    const dialog = document.querySelector(".access-dialog");
    const form = dialog?.querySelector("form");
    const dialogRole = dialog?.querySelector("[data-dialog-role]");
    const intentLabel = dialog?.querySelector("[data-intent-label]");
    const intentSelect = dialog?.querySelector("select[name='intent']");
    const submitButton = dialog?.querySelector(".dialog-submit");
    const formError = dialog?.querySelector("[data-form-error]");
    if (!dialog || !form || !intentSelect || !submitButton || !formError) return;

    let selectedRole = "buyer";

    function setFormError(message = "") {
      formError.textContent = message;
      formError.hidden = !message;
    }

    function configureRole(role) {
      const options = ROLE_OPTIONS[role];
      selectedRole = role;
      if (dialogRole) dialogRole.textContent = options.label;
      if (intentLabel) intentLabel.textContent = options.intentLabel;
      intentSelect.replaceChildren(
        ...options.intents.map((intent) => {
          const option = document.createElement("option");
          option.value = intent;
          option.textContent = intent;
          return option;
        }),
      );
      setFormError();
    }

    document.querySelectorAll(".open-dialog").forEach((button) => {
      button.addEventListener("click", () => {
        configureRole(button.dataset.roleLabel === "节点方" ? "supplier" : "buyer");
        dialog.showModal();
      });
    });

    dialog.querySelector(".dialog-close")?.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      setFormError();
      if (!form.reportValidity()) return;

      const formData = new FormData(form);
      const payload = {
        role: selectedRole,
        name: String(formData.get("name") || ""),
        email: String(formData.get("email") || ""),
        intent: String(formData.get("intent") || ""),
        consent: formData.get("consent") === "on",
      };

      submitButton.disabled = true;
      submitButton.setAttribute("aria-busy", "true");
      const originalLabel = submitButton.innerHTML;
      submitButton.textContent = "正在安全提交…";

      try {
        const result = await requestJson("early-access", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        dialog.close();
        form.reset();
        showToast(
          result.created
            ? "申请已安全提交，我们会通过邮箱联系你。"
            : "申请信息已更新，我们仍会通过邮箱联系你。",
        );
      } catch (error) {
        setFormError(error.message || "提交失败，请稍后再试");
      } finally {
        submitButton.disabled = false;
        submitButton.removeAttribute("aria-busy");
        submitButton.innerHTML = originalLabel;
      }
    });
  }

  setupServiceStatus();
  setupNetworkTilt();
  setupCodeTabs();
  setupRoleTabs();
  setupAccessDialog();
})();
