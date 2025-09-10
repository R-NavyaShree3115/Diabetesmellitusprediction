document.addEventListener("DOMContentLoaded", () => {
  emailjs.init("SwKDSVz7hy2cmcssS"); // Your EmailJS public key

  const tabs = document.querySelectorAll(".tab");
  const subscribeBtn = document.getElementById("subscribeBtn");
  const forms = document.querySelectorAll(".form");
  const secureBadge = document.getElementById("secureBadge");

  activateTab(document.querySelector(".tab.active"));
  tabs.forEach(tab => tab.addEventListener("click", () => activateTab(tab)));
  subscribeBtn.addEventListener("click", () => activateTab(subscribeBtn));

  forms.forEach(form => {
    form.addEventListener("submit", e => {
      e.preventDefault();
      clearFeedback(form);

      if (form.id === "form-doctor") return handleDoctorLogin(form);
      if (form.id === "form-subscribe") return handleSubscription(form);
      if (form.id === "form-premium") return handlePremiumLogin(form);
      if (form.id === "form-patient") return handlePatientLogin(form); // Patient login
      if (form.id === "doctor-pwd-form") return handleGenerateCredentials(form);

      validateAndSuccess(form);
    });

    form.querySelectorAll("input, select").forEach(input => {
      input.addEventListener("input", () => clearError(input));
      input.addEventListener("change", () => clearError(input));
    });
  });

  function activateTab(tabButton) {
    tabs.forEach(t => t.classList.remove("active"));
    subscribeBtn.classList.remove("active");
    forms.forEach(f => f.classList.remove("active"));
    forms.forEach(clearFeedback);

    if (tabButton === subscribeBtn) {
      subscribeBtn.classList.add("active");
      forms.forEach(f => {
        if (f.id === "form-subscribe") f.classList.add("active");
      });
      secureBadge.style.display = "none";
    } else {
      tabButton.classList.add("active");
      const targetForm = document.getElementById(`form-${tabButton.dataset.tab}`);
      if (targetForm) {
        targetForm.classList.add("active");
      }
      secureBadge.style.display = tabButton.dataset.tab === "doctor" ? "inline-block" : "none";
    }
  }

  function validateAndSuccess(form) {
    const inputs = Array.from(form.querySelectorAll("input, select"));
    let allValid = true;
    inputs.forEach(input => {
      if (!input.checkValidity()) {
        allValid = false;
        showError(input, input.validationMessage);
      }
    });
    if (allValid) {
      showSuccess(form, "Login successful");
      form.reset();
    }
  }

  function showError(input, message) {
    input.classList.add("invalid");
    let error = input.parentElement.querySelector(".error-message");
    if (!error) {
      error = document.createElement("div");
      error.classList.add("error-message");
      input.parentElement.appendChild(error);
    }
    error.textContent = message;
  }

  function clearError(input) {
    input.classList.remove("invalid");
    const error = input.parentElement.querySelector(".error-message");
    if (error) error.remove();
  }

  function clearFeedback(form) {
    form.querySelectorAll(".error-message").forEach(el => el.remove());
    form.querySelectorAll("input.invalid, select.invalid").forEach(input => input.classList.remove("invalid"));
    const success = form.querySelector(".success-message");
    if (success) success.remove();
  }

  function showSuccess(form, message) {
    const success = document.createElement("div");
    success.classList.add("success-message");
    success.textContent = message;
    form.appendChild(success);
  }

  // Doctor login handler
  function handleDoctorLogin(form) {
    const email = document.getElementById("doctorEmail");
    const password = document.getElementById("doctorPassword");
    let valid = true;
    [email, password].forEach(input => {
      if (!input.checkValidity()) {
        showError(input, input.validationMessage);
        valid = false;
      }
    });
    if (!valid) return;

    if (email.value === "doctorann2356@gmail.com" && password.value === "ann@123456") {
      showSuccess(form, "Doctor login successful!");
      form.reset();
    } else {
      alert("Invalid doctor credentials.");
    }
  }

  // Subscription form handler
  function handleSubscription(form) {
    const email = document.getElementById("subscribeEmail");  // Updated here to subscribeEmail
    const plan = document.getElementById("subscriptionPlan");
    const payment = document.getElementById("paymentMode");
    const paymentConfirmed = document.getElementById("paymentConfirmed");
    let valid = true;

    [email, plan, payment, paymentConfirmed].forEach(input => {
      if (!input || (input.type === "checkbox" ? !input.checked : !input.checkValidity())) {
        if(input) {
          if (input.type === "checkbox") {
            showError(input, "You must confirm the payment");
          } else {
            showError(input, input.validationMessage);
          }
        }
        valid = false;
      }
    });

    if (!valid) return;

    const templateParams = {
      user_email: email.value,
      patient_name: form.querySelector("#patientName").value,
      patient_contact: form.querySelector("#patientContact").value,
      plan: plan.options[plan.selectedIndex].text,
      payment_mode: payment.value
    };

    emailjs.send("service_8thdwjt", "template_ee1p289", templateParams)
      .then(response => {
        localStorage.setItem("isSubscribed", "true");
        localStorage.setItem("subscribedEmail", email.value.trim());
        showSuccess(form, "Subscription successful! Confirmation sent to your email.");
        form.reset();
      })
      .catch(() => alert("Failed to send confirmation email."));
  }

  // Premium login handler
  function handlePremiumLogin(form) {
    const email = document.getElementById("premiumEmail");
    const password = document.getElementById("premiumPassword");
    let valid = true;
    [email, password].forEach(input => {
      if (!input.checkValidity()) {
        showError(input, input.validationMessage);
        valid = false;
      }
    });
    if (!valid) return;

    const isSubscribed = localStorage.getItem("isSubscribed") === "true";
    const subscribedEmail = localStorage.getItem("subscribedEmail");
    if (!isSubscribed) return alert("Access denied. Please subscribe first.");
    if (email.value.trim() !== subscribedEmail) return alert("Email does not match subscribed account.");

    showSuccess(form, "Premium login successful!");
    form.reset();
  }

  // Patient login handler
  function handlePatientLogin(form) {
    const emailInput = form.querySelector("#patientEmail");
    const email = emailInput.value.trim();

    if (!email) {
      showError(emailInput, "Please enter your email");
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      showError(emailInput, "Please enter a valid email");
      return;
    }

    const templateParams = { patient_email: email };

    emailjs.send("service_8thdwjt", "template_kx9ic7f", templateParams)
      .then(() => {
        showSuccess(form, `Patient login successful! Confirmation sent to ${email}`);
        form.reset();
      })
      .catch(() => {
        alert("Failed to send confirmation email.");
      });
  }

  // Function to send premium credentials via EmailJS (doctor generates)
  function sendPremiumCredentials(patientEmail, premiumEmail, premiumPassword) {
    const templateParams = { patientEmail, premiumEmail, premiumPassword };
    emailjs.send("service_8thdwjt", "template_kx9ic7f", templateParams)
      .then(res => console.log("Credentials emailed:", res.status, res.text))
      .catch(err => console.error("Failed to send credentials:", err));
  }

  // Handle doctor generating patient credentials
  function handleGenerateCredentials(form) {
    const patientEmail = form.querySelector("#patientEmailGen").value.trim();
    const premiumEmail = form.querySelector("#generatedEmail").value.trim();
    const premiumPassword = form.querySelector("#generatedPass").value.trim();

    sendPremiumCredentials(patientEmail, premiumEmail, premiumPassword);
    showSuccess(form, "Premium credentials sent to patient!");
    form.reset();
  }
});
