// ====================
// BMI Calculation
// ====================
function calculateBMI() {
    const weight = parseFloat(document.getElementById("weight").value);
    const height = parseFloat(document.getElementById("height").value);

    if (weight > 0 && height > 0) {
        const heightInMeters = height / 100;
        const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(2);
        document.getElementById("bmi").value = bmi;
    } else {
        document.getElementById("bmi").value = "";
    }
}

// ====================
// Backend ML Prediction
// ====================
async function predictDiabetesBackend() {
    const age = parseInt(document.getElementById("age").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const height = parseFloat(document.getElementById("height").value);
    const fbs = parseFloat(document.getElementById("fbs").value);
    const ppbs = parseFloat(document.getElementById("ppbs").value);

    if (!age || !weight || !height || !fbs || !ppbs) {
        alert("⚠️ Please fill all fields.");
        return;
    }

    // Calculate BMI
    const heightInMeters = height / 100;
    const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(2);
    document.getElementById("bmi").value = bmi;

    const data = {
  Pregnancies: parseInt(document.getElementById("Pregnancies").value),
  Glucose: Glucose,
  BloodPressure: parseFloat(document.getElementById("BloodPressure").value),
  SkinThickness: parseFloat(document.getElementById("SkinThickness").value),
  Insulin: Insulin,
  BMI: parseFloat(document.getElementById("bmi").value),
  DiabetesPedigreeFunction: parseFloat(document.getElementById("DiabetesPedigreeFunction").value),
  Age: parseInt(document.getElementById("Age").value)
};


const response = await fetch("http://127.0.0.1:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data)
});

    try {
      
        if (!response.ok) throw new Error("Server not responding");
        const result = await response.json();
        showResult(result.ensemble_model);

    } catch (err) {
        console.error("❌ Error connecting to backend:", err);
        alert("⚠️ Could not connect to Flask backend. Make sure it's running.");
    }
}
// link this to app.py

// ====================
// Display Result
// ====================
function showResult(status) {
    const healthStatus = document.getElementById("risk-status");
    const healthAdvice = document.getElementById("risk-advice");
    const progressBar = document.getElementById("progress");

    if (!healthStatus || !healthAdvice || !progressBar) return;

    
    if (status === "No Diabetes") {
        progressBar.style.width = "30%";
        progressBar.style.backgroundColor = "green";
        healthAdvice.textContent = "You are healthy! Maintain a good lifestyle.";
    } else if (status === "Prediabetes") {
        progressBar.style.width = "65%";
        progressBar.style.backgroundColor = "orange";
        healthAdvice.textContent = "Risk detected! Start managing diet and exercise.";
    } else {
        progressBar.style.width = "100%";
        progressBar.style.backgroundColor = "red";
        healthAdvice.textContent = "Consult a doctor immediately!";
    }

    // Store latest prediction
    localStorage.setItem("latestPrediction", status);
}

// ---------------------------
// Attach to button click
// ---------------------------
document.getElementById("predict-btn").addEventListener("click", predictDiabetesBackend);


// ====================
// Save Prediction History
// ====================
function savePredictionHistory(status) {
    const today = new Date().toLocaleDateString();
    const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    history.unshift({ status, date: today });
    if (history.length > 5) history.pop();
    localStorage.setItem("predictionHistory", JSON.stringify(history));
    displayHistory();
}

// ====================
// Display History
// ====================
function displayHistory() {
    const activityDiv = document.getElementById("last-prediction");
    const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    if (!activityDiv) return;

    if (history.length === 0) {
        activityDiv.textContent = "No prediction data available.";
    } else {
        activityDiv.innerHTML = history
            .map(item => `<p>${item.date}: <strong>${item.status}</strong></p>`)
            .join("");
    }
}

// ====================
// Load Diet & Exercise Plans
// ====================
function loadHealthPlan() {
    const dietList = document.getElementById("diet-list");
    const exerciseList = document.getElementById("exercise-list");
    if (!dietList || !exerciseList) return;

    const age = parseInt(localStorage.getItem("userAge") || "30");
    const gender = (localStorage.getItem("userGender") || "Male").toLowerCase();
    const prediction = (localStorage.getItem("latestPrediction") || "No Diabetes").toLowerCase();

    // Load Diet CSV
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
        }
    });

    // Load Exercise CSV
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
        }
    });
}

// ====================
// DOMContent Loaded
// ====================
document.addEventListener("DOMContentLoaded", function () {
    displayHistory();
    loadHealthPlan();
});
