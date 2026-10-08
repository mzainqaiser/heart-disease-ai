/* HeartGuard AI - client-side helpers: instant validation, sample values, loading state, gauge animation.
   The server re-validates everything, so this is purely for convenience. */
(function () {
  "use strict";

  // ---- Result page: animate the gauge once the page is visible
  var gauge = document.querySelector(".gauge");
  if (gauge) {
    window.requestAnimationFrame(function () {
      window.setTimeout(function () { gauge.classList.add("animate"); }, 80);
    });
  }

  var form = document.getElementById("predict-form");
  if (!form) { return; }

  var submitBtn = document.getElementById("submit-btn");

  function fieldWrapper(el) { return el.closest(".field"); }

  function messageFor(el) {
    var v = el.validity;
    var label = (fieldWrapper(el).querySelector("label, legend") || {}).textContent || "This field";
    label = label.replace(/\s*\(.*\)\s*$/, "").trim();
    if (v.valueMissing) { return el.type === "radio" ? "Please choose an option." : "This field is required."; }
    if (v.badInput) { return "Please enter a valid number."; }
    if (v.rangeUnderflow || v.rangeOverflow) {
      return label + " must be between " + el.min + " and " + el.max + ".";
    }
    if (el.dataset.integer === "true" && el.value !== "" && !/^-?\d+$/.test(el.value.trim())) {
      return "Please enter a whole number.";
    }
    if (v.stepMismatch) { return "Please use steps of " + el.step + "."; }
    return "";
  }

  function setError(wrapper, message, firstControl) {
    var out = wrapper.querySelector(".field-error");
    out.textContent = message;
    wrapper.classList.toggle("has-error", !!message);
    wrapper.querySelectorAll("input, select").forEach(function (c) {
      if (message) { c.setAttribute("aria-invalid", "true"); } else { c.removeAttribute("aria-invalid"); }
    });
    return firstControl;
  }

  function validateField(el) {
    var wrapper = fieldWrapper(el);
    var control = el.type === "radio" ? wrapper.querySelector("input[type=radio]") : el;
    var msg = "";
    if (el.type === "radio") {
      var checked = wrapper.querySelector("input[type=radio]:checked");
      if (!checked) { msg = "Please choose an option."; }
    } else {
      msg = messageFor(el);
    }
    setError(wrapper, msg);
    return msg ? control : null;
  }

  // Validate on blur / change, and clear an error as soon as the user fixes the value
  form.querySelectorAll("input[type=number], select").forEach(function (el) {
    el.addEventListener("blur", function () { validateField(el); });
    el.addEventListener("input", function () { if (fieldWrapper(el).classList.contains("has-error")) { validateField(el); } });
    el.addEventListener("change", function () { validateField(el); });
  });
  form.querySelectorAll("input[type=radio]").forEach(function (el) {
    el.addEventListener("change", function () { validateField(el); });
  });

  form.addEventListener("submit", function (event) {
    var firstInvalid = null;
    var seenRadio = {};
    form.querySelectorAll("input[type=number], select, input[type=radio]").forEach(function (el) {
      if (el.type === "radio") {
        if (seenRadio[el.name]) { return; }
        seenRadio[el.name] = true;
      }
      var bad = validateField(el);
      if (bad && !firstInvalid) { firstInvalid = bad; }
    });
    if (firstInvalid) {
      event.preventDefault();
      firstInvalid.focus({ preventScroll: true });
      firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    submitBtn.classList.add("is-loading");
    submitBtn.setAttribute("aria-busy", "true");
  });

  // ---- Sample values
  var examples = {};
  try { examples = JSON.parse(form.getAttribute("data-examples") || "{}"); } catch (e) { examples = {}; }

  function fill(values) {
    Object.keys(values).forEach(function (name) {
      var v = String(values[name]);
      var radios = form.querySelectorAll('input[type=radio][name="' + name + '"]');
      if (radios.length) {
        radios.forEach(function (r) { r.checked = (r.value === v); });
      } else {
        var el = form.elements[name];
        if (el) { el.value = v; }
      }
    });
    form.querySelectorAll(".field").forEach(function (w) { setError(w, ""); });
  }

  document.querySelectorAll("[data-example]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var data = examples[btn.getAttribute("data-example")];
      if (data) { fill(data); }
    });
  });

  var reset = document.getElementById("reset-btn");
  if (reset) {
    reset.addEventListener("click", function () {
      window.setTimeout(function () {
        form.querySelectorAll(".field").forEach(function (w) { setError(w, ""); });
      }, 0);
    });
  }

  // Browser back button: make sure the button is not stuck in the loading state
  window.addEventListener("pageshow", function () { submitBtn.classList.remove("is-loading"); submitBtn.removeAttribute("aria-busy"); });
})();
