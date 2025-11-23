SELECT * FROM diabetes_db.patient;

SELECT * FROM diabetes_db.glucose_tracking;
DELETE FROM diabetes_db.glucose_tracking;
DELETE FROM diabetes_db.bp_tracking;
SELECT * FROM diabetes_db.bp_tracking;

TRUNCATE TABLE diabetes_db.glucose_tracking;
TRUNCATE TABLE diabetes_db.bp_tracking;

ALTER TABLE glucose_tracking
ADD COLUMN glucose FLOAT AFTER patient_id,
ADD COLUMN insulin FLOAT AFTER glucose,
ADD COLUMN dpf FLOAT AFTER insulin,
ADD COLUMN age INT AFTER dpf,
ADD COLUMN outcome INT AFTER age,
ADD COLUMN reading_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP AFTER outcome;

