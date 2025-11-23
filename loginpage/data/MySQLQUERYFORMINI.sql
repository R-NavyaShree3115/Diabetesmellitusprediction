SELECT * FROM diabetes_db.patient;

SELECT * FROM diabetes_db.glucose_tracking;
DELETE FROM diabetes_db.glucose_tracking;
DELETE FROM diabetes_db.bp_tracking;
SELECT * FROM diabetes_db.bp_tracking;

TRUNCATE TABLE diabetes_db.glucose_tracking;
TRUNCATE TABLE diabetes_db.bp_tracking;
