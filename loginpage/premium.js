// ====================
// BMI Calculation
// ====================
function calculateBMI() {
    const weight = parseFloat(document.getElementById("weight").value);
    const height = parseFloat(document.getElementById("height").value);

    if (!weight || !height) {
        alert("Please enter valid weight and height.");
        return;
    }

    const heightInMeters = height / 100;
    const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(2);

    document.getElementById("bmi").value = bmi;
}

// ====================
// Diabetes Prediction
// ====================
function predictDiabetes() {
    const age = parseInt(document.getElementById("age").value);
    const gender = document.getElementById("gender").value;
    const region = document.getElementById("region").value;
    const weight = parseFloat(document.getElementById("weight").value);
    const height = parseFloat(document.getElementById("height").value);
    const fbs = parseFloat(document.getElementById("fbs").value);
    const ppbs = parseFloat(document.getElementById("ppbs").value);
    const hba1c = parseFloat(document.getElementById("hba1c").value);

    if (!age || !gender || !region || !weight || !height || !fbs || !ppbs || !hba1c) {
        alert("Please fill in all fields before predicting.");
        return;
    }

    const heightInMeters = height / 100;
    const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(2);
    document.getElementById("bmi").value = bmi;

    let riskScore = 0;

    // Age factor
    if (age > 45) riskScore += 1;

    // BMI factor
    if (bmi > 25) riskScore += 1;

// Show result section
document.getElementById("prediction-result").style.display = "block";

// Calculate prediction
let prediction = "Normal";

if (fbs >= 126 || ppbs >= 200 || hba1c >= 6.5) {
    prediction = "Diabetes";
} else if (fbs >= 100 || ppbs >= 140 || hba1c >= 5.7) {
    prediction = "Prediabetes";
}

// Display prediction
document.getElementById("risk-status").textContent = prediction;

const progressBar = document.getElementById("progress");
const riskAdvice = document.getElementById("risk-advice");

// Animate width and color
progressBar.style.width = "0%"; // reset
setTimeout(() => {
    if (prediction === "Normal") {
        progressBar.style.width = "30%";
        progressBar.style.backgroundColor = "#28a745"; // Green
        riskAdvice.textContent = "✅ You are healthy! Maintain a balanced lifestyle.";
    } else if (prediction === "Prediabetes") {
        progressBar.style.width = "65%";
        progressBar.style.backgroundColor = "#ff9800"; // Orange
        riskAdvice.textContent = "⚠️ Risk detected! Start diet & exercise management.";
    } else {
        progressBar.style.width = "100%";
        progressBar.style.backgroundColor = "#dc3545"; // Red
        riskAdvice.textContent = "❗ High risk! Consult a doctor immediately.";
    }
}, 50);


    // Save to localStorage for dashboard
    const today = new Date().toLocaleDateString();
    const predictionEntry = {
        status: prediction,
        date: today
    };

    let history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    history.unshift(predictionEntry);
    if (history.length > 5) history = history.slice(0, 5);
    localStorage.setItem("predictionHistory", JSON.stringify(history));
}
// ====================
// Backend Prediction
// ====================
async function predictDiabetesBackend() {
    const age = parseInt(document.getElementById("age").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const height = parseFloat(document.getElementById("height").value);
    const fbs = parseFloat(document.getElementById("fbs").value);
    const ppbs = parseFloat(document.getElementById("ppbs").value);

    const heightInMeters = height / 100;
    const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(2);
    document.getElementById("bmi").value = bmi;

    const data = {
        Pregnancies: 0, // add input if you want real value
        Glucose: fbs,
        BloodPressure: 80, // default if not taken
        SkinThickness: 20,
        Insulin: ppbs,
        BMI: parseFloat(bmi),
        DiabetesPedigreeFunction: 0.5, // default
        Age: age
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const result = await response.json();

        // Update frontend with backend result
        showPredictionBackend(result);
    } catch (err) {
        console.error("Error connecting to backend:", err);
        alert("Could not connect to server. Make sure Flask is running.");
    }
}

// Display result from backend
function showPredictionBackend(result) {
    document.getElementById("prediction-result").style.display = "block";
    const status = result.label;
    const prob = result.probability ? (result.probability * 100).toFixed(1) : "--";

    document.getElementById("risk-status").textContent = `${status} (${prob}%)`;

    const progressBar = document.getElementById("progress");
    const riskAdvice = document.getElementById("risk-advice");

    progressBar.style.width = prob + "%";
    if (status === "Non-diabetic") {
        progressBar.style.backgroundColor = "#28a745";
        riskAdvice.textContent = "✅ You are healthy! Maintain a balanced lifestyle.";
    } else {
        progressBar.style.backgroundColor = "#dc3545";
        riskAdvice.textContent = "❗ High risk! Consult a doctor immediately.";
    }

    // Save to localStorage
    const today = new Date().toLocaleDateString();
    const predictionEntry = { status, date: today };
    let history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    history.unshift(predictionEntry);
    if (history.length > 5) history = history.slice(0, 5);
    localStorage.setItem("predictionHistory", JSON.stringify(history));
}

// ====================
// Dashboard Logic
// ====================
document.addEventListener("DOMContentLoaded", function () {
    const activityDiv = document.getElementById("last-prediction");
    const healthStatus = document.getElementById("risk-status");
    const healthAdvice = document.getElementById("risk-advice");
    const progressBar = document.getElementById("progress");

    const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];

    // Recent Activity
    if (activityDiv) {
        if (history.length === 0) {
            activityDiv.textContent = "No prediction data available.";
        } else {
            activityDiv.innerHTML = history
                .map(item => `<p>${item.date}: <strong>${item.status}</strong></p>`)
                .join("");
        }
    }

    // Health Summary (latest prediction)
    if (healthStatus && history.length > 0) {
        const latest = history[0];
        healthStatus.textContent = latest.status;

        if (latest.status === "Normal") {
            progressBar.style.width = "30%";
            progressBar.style.backgroundColor = "green";
            healthAdvice.textContent = "You are healthy! Maintain a good lifestyle.";
        } else if (latest.status === "Prediabetes") {
            progressBar.style.width = "65%";
            progressBar.style.backgroundColor = "orange";
            healthAdvice.textContent = "Risk detected! Start managing diet and exercise.";
        } else {
            progressBar.style.width = "100%";
            progressBar.style.backgroundColor = "red";
            healthAdvice.textContent = "Consult a doctor immediately!";
        }
    }
function loadHealthPlan() {
      const dietList = document.getElementById("diet-list");
      const exerciseList = document.getElementById("exercise-list");

      // Get user data from localStorage
      const age = parseInt(localStorage.getItem("userAge") || "30");
      const gender = (localStorage.getItem("userGender") || "Male").toLowerCase();
      const prediction = (localStorage.getItem("latestPrediction") || "Normal").toLowerCase();

      // --- Load Diet CSV ---
      Papa.parse("data/combined_diabetes_datasets.csv", {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: function(results) {
          const rows = results.data;
          const matchedDiet = rows.find(r => {
            const rowCondition = (r.condition || "").trim().toLowerCase();
            const rowSex = (r.sex || "").trim().toLowerCase();
            const rowAge = parseInt(r.age) || 30;
            return rowCondition === prediction && rowSex === gender && Math.abs(rowAge - age) <= 5;
          });

          dietList.innerHTML = "";
          if (matchedDiet && matchedDiet.diet_plan) {
            matchedDiet.diet_plan.split("|").forEach(item => {
              const li = document.createElement("li");
              li.innerHTML = `<i class="fas fa-apple-alt"></i> ${item.trim()}`;
              dietList.appendChild(li);
            });
          } else {
            dietList.innerHTML = "<li>No diet plan found for your profile.</li>";
          }
        },
        error: function(err) {
          console.error("Diet CSV Load Error:", err);
          dietList.innerHTML = "<li>Error loading diet plan.</li>";
        }
      });

      // --- Load Exercise CSV ---
      Papa.parse("data/exercise_dataset.csv", {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: function(results) {
          const rows = results.data;
          const matchedEx = rows.find(r => {
            const rowCondition = (r.condition || "").trim().toLowerCase();
            const rowSex = (r.sex || "").trim().toLowerCase();
            let minAge = 0, maxAge = 120;
            if (r.age_range) {
              const parts = r.age_range.split("-");
              minAge = parseInt(parts[0]);
              maxAge = parseInt(parts[1]);
            }
            return rowCondition === prediction && rowSex === gender && age >= minAge && age <= maxAge;
          });

          exerciseList.innerHTML = "";
          if (matchedEx && matchedEx.exercise_plan) {
            matchedEx.exercise_plan.split(";").forEach(item => {
              const li = document.createElement("li");
              li.innerHTML = `<i class="fas fa-dumbbell"></i> ${item.trim()}`;
              exerciseList.appendChild(li);
            });
            if (matchedEx && matchedEx.yoga_recommendation) {
              const li = document.createElement("li");
              li.innerHTML = `<i class="fas fa-person"></i> Yoga: ${matchedEx.yoga_recommendation}`;
              exerciseList.appendChild(li);
            }
          } else {
            exerciseList.innerHTML = "<li>No exercise plan found for your profile.</li>";
          }
        },
        error: function(err) {
          console.error("Exercise CSV Load Error:", err);
          exerciseList.innerHTML = "<li>Error loading exercise plan.</li>";
        }
      });
    }

    document.addEventListener("DOMContentLoaded", loadHealthPlan);
});
