CREATE DATABASE sperm_db;
USE sperm_db;
SELECT * FROM sperm_selection;
DESCRIBE sperm_selection;
/* =========================================================
   1. BASIC OVERVIEW
   ========================================================= */

-- Total records
SELECT COUNT(*) AS total_records
FROM sperm_selection;

-- Sample data
SELECT *
FROM sperm_selection
LIMIT 10;


/* =========================================================
   2. DATA QUALITY CHECKS
   ========================================================= */

-- NULL count per important column
SELECT
    COUNT(*) - COUNT(Sperm_Concentration_M_per_ml) AS null_sperm_concentration,
    COUNT(*) - COUNT(Total_Motility_Percent) AS null_motility,
    COUNT(*) - COUNT(Selection_Time_Seconds) AS null_selection_time,
    COUNT(*) - COUNT(Motility_Pattern) AS null_motility_pattern
FROM sperm_selection;

-- Duplicate records check
SELECT
    Record_ID,
    Patient_ID,
    Oocyte_ID,
    COUNT(*) AS duplicate_count
FROM sperm_selection
GROUP BY Record_ID, Patient_ID, Oocyte_ID
HAVING COUNT(*) > 1;


/* =========================================================
   3. DESCRIPTIVE STATISTICS (EDA CORE)
   ========================================================= */

SELECT
    MIN(Sperm_Concentration_M_per_ml) AS min_concentration,
    MAX(Sperm_Concentration_M_per_ml) AS max_concentration,
    AVG(Sperm_Concentration_M_per_ml) AS avg_concentration,
    STDDEV(Sperm_Concentration_M_per_ml) AS std_concentration,

    MIN(Total_Motility_Percent) AS min_motility,
    MAX(Total_Motility_Percent) AS max_motility,
    AVG(Total_Motility_Percent) AS avg_motility
FROM sperm_selection;


/* =========================================================
   4. QUARTILES + IQR (OUTLIER DETECTION)
   ========================================================= */

WITH ordered AS (
    SELECT
        Sperm_Concentration_M_per_ml,
        ROW_NUMBER() OVER (ORDER BY Sperm_Concentration_M_per_ml) AS rn,
        COUNT(*) OVER () AS total_rows
    FROM sperm_selection
),
quartiles AS (
    SELECT
        MAX(CASE WHEN rn = FLOOR(0.25 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS Q1,
        MAX(CASE WHEN rn = FLOOR(0.75 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS Q3
    FROM ordered
)
SELECT
    Q1,
    Q3,
    (Q1 - 1.5 * (Q3 - Q1)) AS lower_limit,
    (Q3 + 1.5 * (Q3 - Q1)) AS upper_limit
FROM quartiles;
-- OUTLIERS
WITH ordered AS (
    SELECT
        Sperm_Concentration_M_per_ml,
        ROW_NUMBER() OVER (ORDER BY Sperm_Concentration_M_per_ml) AS rn,
        COUNT(*) OVER () AS total_rows
    FROM sperm_selection
),
quartiles AS (
    SELECT
        MAX(CASE WHEN rn = FLOOR(0.25 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS Q1,
        MAX(CASE WHEN rn = FLOOR(0.75 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS Q3
    FROM ordered
),
bounds AS (
    SELECT
        Q1,
        Q3,
        (Q1 - 1.5 * (Q3 - Q1)) AS lower_limit,
        (Q3 + 1.5 * (Q3 - Q1)) AS upper_limit
    FROM quartiles
)
SELECT
    s.*
FROM sperm_selection s
CROSS JOIN bounds b
WHERE
    s.Sperm_Concentration_M_per_ml < b.lower_limit
    OR
    s.Sperm_Concentration_M_per_ml > b.upper_limit;


/* =========================================================
   5. CATEGORICAL DISTRIBUTION
   ========================================================= */

SELECT
    Motility_Pattern,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM sperm_selection
GROUP BY Motility_Pattern
ORDER BY count DESC;


/* =========================================================
   6. DISCRETIZATION (BINNING)
   ========================================================= */

-- Mean based binning
SELECT
    *,
    CASE
        WHEN Sperm_Concentration_M_per_ml <
             (SELECT AVG(Sperm_Concentration_M_per_ml) FROM sperm_selection)
        THEN 'Low'
        ELSE 'High'
    END AS sperm_concentration_level
FROM sperm_selection;

-- Quartile based binning
WITH ordered AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY Sperm_Concentration_M_per_ml) AS rn,
        COUNT(*) OVER () AS total_rows
    FROM sperm_selection
),
quartiles AS (
    SELECT
        MAX(CASE WHEN rn = FLOOR(0.25 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS q1,
        MAX(CASE WHEN rn = FLOOR(0.50 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS q2,
        MAX(CASE WHEN rn = FLOOR(0.75 * total_rows) THEN Sperm_Concentration_M_per_ml END) AS q3
    FROM ordered
)
SELECT
    s.*,
    CASE
        WHEN s.Sperm_Concentration_M_per_ml <= q.q1 THEN 'Low'
        WHEN s.Sperm_Concentration_M_per_ml <= q.q2 THEN 'Medium'
        WHEN s.Sperm_Concentration_M_per_ml <= q.q3 THEN 'High'
        ELSE 'Very High'
    END AS sperm_group
FROM sperm_selection s
CROSS JOIN quartiles q;


/* =========================================================
   7. PERFORMANCE EDA (MOST IMPORTANT)
   ========================================================= */

-- Fertilization Success vs Features
SELECT
    Fertilization_Success,
    COUNT(*) AS samples,
    AVG(Sperm_Concentration_M_per_ml) AS avg_concentration,
    AVG(Total_Motility_Percent) AS avg_motility,
    AVG(Selection_Time_Seconds) AS avg_selection_time
FROM sperm_selection
GROUP BY Fertilization_Success;

-- Usable Embryo vs Sperm Quality
SELECT
    Usable_Embryo,
    COUNT(*) AS cases,
    AVG(Sperm_Concentration_M_per_ml) AS avg_concentration,
    AVG(Total_Motility_Percent) AS avg_motility
FROM sperm_selection
GROUP BY Usable_Embryo;


/* =========================================================
   8. CORRELATION (PostgreSQL)
   ========================================================= */

SELECT
(
    COUNT(*) * SUM(Sperm_Concentration_M_per_ml * Total_Motility_Percent)
    - SUM(Sperm_Concentration_M_per_ml) * SUM(Total_Motility_Percent)
)
/
SQRT(
    (COUNT(*) * SUM(POW(Sperm_Concentration_M_per_ml, 2))
        - POW(SUM(Sperm_Concentration_M_per_ml), 2))
    *
    (COUNT(*) * SUM(POW(Total_Motility_Percent, 2))
        - POW(SUM(Total_Motility_Percent), 2))
) AS corr_concentration_motility
FROM sperm_selection;

/* =========================================================
   9. MISSING VALUE IMPUTATION (OPTIONAL)
   ========================================================= */

-- Mean Imputation
SET SQL_SAFE_UPDATES = 0;

-- 2. Mean imputation
UPDATE sperm_selection
SET Sperm_Concentration_M_per_ml = (
    SELECT avg_val
    FROM (
        SELECT AVG(Sperm_Concentration_M_per_ml) AS avg_val
        FROM sperm_selection
    ) AS t
)
WHERE Sperm_Concentration_M_per_ml IS NULL;

-- 3. (Optional) Re-enable safe updates
SET SQL_SAFE_UPDATES = 1;

UPDATE sperm_selection
SET Motility_Pattern = (
    SELECT mode_val
    FROM (
        SELECT Motility_Pattern AS mode_val
        FROM sperm_selection
        WHERE Motility_Pattern IS NOT NULL
        GROUP BY Motility_Pattern
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ) AS t
)
WHERE Motility_Pattern IS NULL;

SET SQL_SAFE_UPDATES = 1;

/* =========================================================
   10. FINAL DATASET FOR DASHBOARD / MODEL
   ========================================================= */

CREATE OR REPLACE VIEW sperm_eda_summary AS
SELECT
    Fertilization_Success,
    Usable_Embryo,
    AVG(Sperm_Concentration_M_per_ml) AS avg_concentration,
    AVG(Total_Motility_Percent) AS avg_motility,
    COUNT(*) AS records
FROM sperm_selection
GROUP BY Fertilization_Success, Usable_Embryo;