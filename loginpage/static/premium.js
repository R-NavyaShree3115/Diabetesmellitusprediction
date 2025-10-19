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

// ==============================
// DASHBOARD LOGIC
// ==============================
document.addEventListener("DOMContentLoaded", function () {
  const riskStatus = document.getElementById("risk-status");
  const riskAdvice = document.getElementById("risk-advice");
  const progressBar = document.getElementById("progress");
  const historyDiv = document.getElementById("last-prediction");

  if (!riskStatus) return; // Skip if not on dashboard.html

  const latest = localStorage.getItem("latestPrediction");
  const advice = localStorage.getItem("latestAdvice");

  if (latest) {
    const isDiabetic = latest.toLowerCase().includes("diabetic");
    const color = isDiabetic ? "red" : "green";
    const width = isDiabetic ? "85%" : "35%";

    riskStatus.innerHTML = `<strong>Status:</strong> <span style="color:${color};">${latest}</span>`;
    progressBar.style.width = width;
    progressBar.style.backgroundColor = color;
    riskAdvice.innerHTML = `<strong>Advice:</strong> ${advice || "Maintain a healthy lifestyle."}`;
  } else {
    riskStatus.textContent = "No prediction data found. Please make a prediction first.";
    progressBar.style.width = "0";
    riskAdvice.textContent = "";
  }

  // Show prediction history
  displayHistory();

  // Load health plan dynamically
  loadHealthPlan();
});

// ==============================
// HEALTH PLAN LOADER
// ==============================
function loadHealthPlan() {
  const dietList = document.getElementById("diet-list");
  const exerciseList = document.getElementById("exercise-list");
  if (!dietList || !exerciseList) return;

  const age = parseInt(localStorage.getItem("userAge") || "30");
  const gender = (localStorage.getItem("userGender") || "female").toLowerCase();
  const prediction = (localStorage.getItem("latestPrediction") || "non-diabetic").toLowerCase();

  // Load Diet CSV
  Papa.parse("/data/combined_diabetes_datasets.csv", {
    download: true,
    header: true,
    complete: function (results) {
      const match = results.data.find(r =>
        (r.condition || "").trim().toLowerCase() === prediction &&
        (r.sex || "").trim().toLowerCase() === gender
      );

      dietList.innerHTML = "";
      if (match && match.diet_plan) {
        match.diet_plan.split("|").forEach(item => {
          const li = document.createElement("li");
          li.innerHTML = `<i class="fas fa-apple-alt"></i> ${item.trim()}`;
          dietList.appendChild(li);
        });
      } else {
        dietList.innerHTML = "<li>No diet plan found.</li>";
      }
    }
  });

  // Load Exercise CSV
  Papa.parse("/data/exercise_dataset.csv", {
    download: true,
    header: true,
    complete: function (results) {
      const match = results.data.find(r =>
        (r.condition || "").trim().toLowerCase() === prediction &&
        (r.sex || "").trim().toLowerCase() === gender
      );

      exerciseList.innerHTML = "";
      if (match && match.exercise_plan) {
        match.exercise_plan.split(";").forEach(item => {
          const li = document.createElement("li");
          li.innerHTML = `<i class="fas fa-dumbbell"></i> ${item.trim()}`;
          exerciseList.appendChild(li);
        });
      } else {
        exerciseList.innerHTML = "<li>No exercise plan found.</li>";
      }
    }
  });
}