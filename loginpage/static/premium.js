document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("patientForm");
  const predictionDiv = document.getElementById("prediction");
  const adviceDiv = document.getElementById("advice");
  const resultSection = document.getElementById("result-section");
  const predictBtn = document.getElementById("predict-btn");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Disable button while predicting
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

      // Show result section dynamically
      resultSection.style.display = "block";

      if (result.error) {
        predictionDiv.innerHTML = `<span style="color:red;">⚠️ Error: ${result.error}</span>`;
        adviceDiv.innerHTML = "";
      } else {
        const color = result.prediction === 1 ? "red" : "green";
        const text = result.prediction === 1 ? "🩸 Diabetic" : "💚 Non-Diabetic";
        predictionDiv.innerHTML = `<strong>Prediction:</strong> <span style="color:${color};">${text}</span>`;
        adviceDiv.innerHTML = `<strong>Advice:</strong> ${result.advice}`;
      }

      // Smooth scroll to result
      resultSection.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      predictionDiv.innerHTML = `<span style="color:red;">⚠️ Could not connect to server.</span>`;
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerText = "📊 Save & Predict";
    }
  });
});

