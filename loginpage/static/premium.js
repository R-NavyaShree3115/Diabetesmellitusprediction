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


// ==============================
// HISTORY MANAGEMENT
// ==============================
function updateHistory(latest) {
  const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
  const today = new Date().toLocaleDateString();

  history.unshift({ status: latest, date: today });
  if (history.length > 5) history.pop();

  localStorage.setItem("predictionHistory", JSON.stringify(history));
}

function displayHistory() {
  const historyDiv = document.getElementById("last-prediction");
  if (!historyDiv) return;

  const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
  if (history.length === 0) {
    historyDiv.textContent = "No prediction data available.";
  } else {
    historyDiv.innerHTML = history
      .map(item => `<p>${item.date}: <strong>${item.status}</strong></p>`)
      .join("");
  }
}

 // ======== DASHBOARD LOGIC (dashboard.html) ========
  const riskStatus = document.getElementById("risk-status");
  const riskAdvice = document.getElementById("risk-advice");
  const progressBar = document.getElementById("progress");
  const historyDiv = document.getElementById("prediction-history");
  const recommendationsDiv = document.getElementById("recommendations");

  if (riskStatus && historyDiv) { // Only run on dashboard.html
    const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    const latest = localStorage.getItem("latestPrediction") || null;
    const advice = localStorage.getItem("latestAdvice") || "Maintain a balanced diet and exercise.";

    // Health Summary
    if (latest) {
      const isDiabetic = latest.includes("Diabetic");
      const color = isDiabetic ? "red" : "green";
      const width = isDiabetic ? "85%" : "35%";
      riskStatus.innerHTML = `<strong>Status:</strong> <span style="color:${color}">${latest}</span>`;
      progressBar.style.width = width;
      progressBar.style.backgroundColor = color;
      riskAdvice.innerHTML = `<strong>Advice:</strong> ${advice}`;
    } else {
      riskStatus.textContent = "No prediction data found.";
      progressBar.style.width = "0";
      riskAdvice.textContent = "";
    }

    // Prediction History
    if (history.length === 0) {
      historyDiv.textContent = "No previous predictions available.";
    } else {
      historyDiv.innerHTML = history
        .map(h => `<p>${h.date}: <strong>${h.status}</strong></p>`)
        .join("");
    }

    // Charts
    const ctxPie = document.getElementById("riskPieChart")?.getContext("2d");
    const ctxGlucose = document.getElementById("glucoseChart")?.getContext("2d");
    const ctxAdherence = document.getElementById("adherenceChart")?.getContext("2d");

    // Pie chart: Diabetic vs Non-Diabetic
    if (ctxPie) {
      const diabeticCount = history.filter(h => h.status.includes("Diabetic")).length;
      const nonDiabeticCount = history.filter(h => h.status.includes("Non-Diabetic")).length;
      new Chart(ctxPie, {
        type: "pie",
        data: {
          labels: ["Diabetic", "Non-Diabetic"],
          datasets: [{ data: [diabeticCount, nonDiabeticCount], backgroundColor: ["#ff4d4d", "#4CAF50"] }]
        },
        options: { plugins: { title: { display: true, text: "Prediction Distribution" } } }
      });
    }

    // Line chart: Glucose trend
    if (ctxGlucose) {
      const dates = history.map(h => h.date).reverse();
      const glucoseValues = history.map(h => h.glucose || 100).reverse();
      new Chart(ctxGlucose, {
        type: "line",
        data: { labels: dates, datasets: [{ label: "Glucose Trend", data: glucoseValues, borderColor: "#36A2EB", fill: true, backgroundColor: "rgba(54,162,235,0.3)" }] }
      });
    }

    // Bar chart: Diet & Exercise adherence
    if (ctxAdherence) {
      const dietAdherence = history.map(h => h.dietAdherence || 0).reverse();
      const exerciseAdherence = history.map(h => h.exerciseAdherence || 0).reverse();
      new Chart(ctxAdherence, {
        type: "bar",
        data: {
          labels: history.map(h => h.date).reverse(),
          datasets: [
            { label: "Diet %", data: dietAdherence, backgroundColor: "#4CAF50" },
            { label: "Exercise %", data: exerciseAdherence, backgroundColor: "#FF9800" }
          ]
        },
        options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
      });
    }

    // Recommendations based on latest risk
    if (history.length > 0) {
      const lastRisk = history[0].status.includes("Diabetic") ? "high" : "low";
      recommendationsDiv.innerHTML = lastRisk === "high"
        ? "<ul><li>Increase physical activity.</li><li>Follow strict diet plan.</li><li>Consult doctor for next check-up.</li></ul>"
        : "<ul><li>Maintain current diet and exercise routine.</li><li>Regular monitoring recommended.</li></ul>";
    } else {
      recommendationsDiv.textContent = "No recommendations available yet. Make your first prediction!";
    }
  }

