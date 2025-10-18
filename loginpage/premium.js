// ====================
// Backend ML Prediction
// ====================
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("patientForm");
    const resultSection = document.getElementById("prediction-result");
    const resultMessage = document.getElementById("resultMessage");

    form.addEventListener("submit", async function(e) {
        e.preventDefault();

        // Collect input values
        const data = {
            Age: parseInt(document.getElementById("Age").value),
            Gender: document.getElementById("Gender").value,
            Pregnancies: parseInt(document.getElementById("Pregnancies").value),
            Glucose: parseFloat(document.getElementById("Glucose").value),
            BloodPressure: parseFloat(document.getElementById("BloodPressure").value),
            SkinThickness: parseFloat(document.getElementById("SkinThickness").value),
            Insulin: parseFloat(document.getElementById("Insulin").value),
            BMI: parseFloat(document.getElementById("BMI").value),
            DiabetesPedigreeFunction: parseFloat(document.getElementById("DiabetesPedigreeFunction").value)
        };

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await res.json();

            if(result.status === "success") {
                resultSection.style.display = "block";
                resultMessage.innerHTML = `<h3>Prediction: ${result.ensemble_model}</h3>`;
            } else {
                alert("⚠️ Prediction failed: " + result.message);
            }

        } catch (err) {
            console.error(err);
            alert("⚠️ Could not connect to Flask backend. Make sure it's running.");
        }
    });
});


// ====================
// Display Result
// ====================
function showResult(status) {
    const healthStatus = document.getElementById("risk-status");
    const healthAdvice = document.getElementById("risk-advice");
    const progressBar = document.getElementById("progress");

    if (!healthStatus || !healthAdvice || !progressBar) return;

    healthStatus.textContent = status;

    if (status === "Non-Diabetic") {
        progressBar.style.width = "30%";
        progressBar.style.backgroundColor = "green";
        healthAdvice.textContent = "You are healthy! Maintain a good lifestyle.";
    } else {
        progressBar.style.width = "100%";
        progressBar.style.backgroundColor = "red";
        healthAdvice.textContent = "Consult a doctor immediately!";
    }

    // Store latest prediction
    localStorage.setItem("latestPrediction", status);
}

// ====================
// Attach to button click
// ====================
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
