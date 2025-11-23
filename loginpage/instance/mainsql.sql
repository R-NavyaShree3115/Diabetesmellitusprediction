-- -----------------------------------------------------------
-- MySQL Schema for Diabetes Prediction and Health Plan Application
-- -----------------------------------------------------------

-- 1. Database Setup: Drop/Recreate for a clean start
DROP DATABASE IF EXISTS diabetes_db;
CREATE DATABASE diabetes_db;
USE diabetes_db;

-- 2. Drop Tables (in dependency order)
-- Dropping tables must respect foreign key constraints.
DROP TABLE IF EXISTS HealthPlan;
DROP TABLE IF EXISTS glucose_tracking;
DROP TABLE IF EXISTS doctor;
DROP TABLE IF EXISTS patient;

-- 3. Table: patient
-- Stores user login credentials and all baseline PIMA features.
-- Features are nullable to allow registration before data entry.
CREATE TABLE patient (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,

    -- PIMA features, all now allowing NULL as per the requested fixes
    pregnancies INT NULL DEFAULT 0,
    glucose INT NULL DEFAULT 100,
    blood_pressure INT NULL DEFAULT 70,
    skin_thickness INT NULL DEFAULT 20,
    insulin INT NULL DEFAULT 0,
    bmi FLOAT NULL DEFAULT 25.0,
    diabetes_pedigree_function FLOAT NULL DEFAULT 0.5,
    age INT NULL DEFAULT 30,

    -- Last prediction outcome (0 = Negative, 1 = Positive)
    last_diabetes_outcome INT NULL DEFAULT 0
);

-- 4. Table: doctor
-- Stores credentials for doctor accounts.
CREATE TABLE doctor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- 5. Table: HealthPlan
-- Stores personalized diet, exercise, and yoga plans for each patient.
CREATE TABLE HealthPlan (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL UNIQUE, -- UNIQUE ensures a 1:1 relationship (one current plan per patient)
    diet_plan TEXT,
    exercise_plan TEXT,
    yoga_plan TEXT,

    -- Link to the patient table
    FOREIGN KEY (patient_id) REFERENCES patient(id) ON DELETE CASCADE
);

-- 6. Table: glucose_tracking
-- Stores historical PIMA feature data used for tracking and/or running predictions.
-- This now acts as the comprehensive prediction history table.
CREATE TABLE glucose_tracking (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL, -- Link back to the patient
    insulin FLOAT,
    glucose FLOAT,
    dpf FLOAT, -- Renamed from diabetes_pedigree_function to dpf for brevity here
    age INT,
    outcome TINYINT, -- 0 or 1 for prediction result at that time
    -- RENAME for clarity: This DATETIME column ensures every reading is recorded separately by time.
    reading_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Link to the patient table
    FOREIGN KEY (patient_id) REFERENCES patient(id) ON DELETE CASCADE
);
DROP TABLE IF EXISTS bp_tracking;

CREATE TABLE bp_tracking (
  id INT AUTO_INCREMENT PRIMARY KEY,
  patient_id INT NOT NULL,
  bp_value FLOAT NOT NULL,
  reading_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patient(id)
);

ALTER TABLE HealthPlan
ADD COLUMN last_generated DATETIME DEFAULT CURRENT_TIMESTAMP;

-- Show structure to verify the final schema
DESCRIBE patient;
DESCRIBE doctor;
DESCRIBE HealthPlan;
DESCRIBE glucose_tracking;
DESCRIBE bp_tracking;