document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("patientForm");
  const predictionDiv = document.getElementById("prediction");
  const adviceDiv = document.getElementById("advice");
  const resultSection = document.getElementById("result-section");
  const predictBtn = document.getElementById("predict-btn");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    predictBtn.disabled = true;
    predictBtn.innerText = "⏳ Predicting...";

    predictionDiv.innerHTML = "<strong>Predicting...</strong>";
    adviceDiv.innerHTML = "";

    const data = {
      Age: parseFloat(document.getElementById("Age").value),
      Pregnancies: parseFloat(document.getElementById("Pregnancies").value),
      Glucose: parseFloat(document.getElementById("Glucose").value),
      BloodPressure: parseFloat(document.getElementById("BloodPressure").value),
      SkinThickness: parseFloat(document.getElementById("SkinThickness").value),
      Insulin: parseFloat(document.getElementById("Insulin").value),
      BMI: parseFloat(document.getElementById("BMI").value),
      DiabetesPedigreeFunction: parseFloat(document.getElementById("DiabetesPedigreeFunction").value)
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      const result = await response.json();

      resultSection.style.display = "block";

      if (result.redirect) {
        window.location.href = result.redirect;
        return;
      }

      if (result.error) {
        predictionDiv.innerHTML = `<span style="color:red;">⚠️ Error: ${result.error}</span>`;
        adviceDiv.innerHTML = "";
      } else {
        const color = result.prediction === 1 ? "red" : "green";
        const text = result.prediction === 1 ? "🩸 Diabetic" : "💚 Non-Diabetic";

        predictionDiv.innerHTML =
          `<strong>Prediction:</strong> <span style="color:${color};">${text}</span>`;

        let advice = "";

        if (result.prediction === 1) {
          advice = "⚠️ High risk of diabetes. Consult doctor. Maintain healthy diet and exercise.";
        } else {
          advice = "✅ Low risk of diabetes. Maintain healthy lifestyle.";
        }

        adviceDiv.innerHTML = `<strong>Advice:</strong> ${advice}`;
      }

      resultSection.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      predictionDiv.innerHTML = `<span style="color:red;">⚠️ Could not connect to server.</span>`;
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerText = "📊 Save & Predict";
    }
  });
});
