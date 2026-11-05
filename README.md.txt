Predicting Emergency Department Triage Acuity from Presenting Complaint Text and Vital Signs: A Hybrid Model with subgroup equity audit (MIMIC-IV-ED)



Applied Artificial Intelligence in Health Care

Name: Hari Bhandari 

Date: 23-10-2026

Table of Contents 
ABSTRACT	4
1.	Background	5
2.	Objectives	5
3.	Introduction	6
4.	Methods	7
Data Preparation	7
Feature Selection and Engineering	8
Model Selection	8
Training and Validation	9
Model Evaluation	10
5.	Results	10
Model Comparison	10
Table 1. Test-set performance of seven models with 83,620 stays and their suitability in New Zealand	12
Figure 1. Default text-only, structured-only and hybrid models: (a) recall by ESI level (plot), (b) macro F1 (bar) and (c) confusion matrix of the default hybrid model (imshow).	12
Tuning	12
Table 2. Hyperparameter search (unweighted macro F1, patient-grouped three-fold cross-validation)	13
Operating point	13
Table 3. Test-set miss rate and false-alarm rate when each hybrid model is set to miss a chosen share of urgent patients.	13
Figure 2. Trade-off curves between missed urgent patients (under-triage) and false alarms for the default and tuned hybrid models, without race. Dots mark each model's own decision rule.	14
Calibration	14
Figure 3. Calibration of the tuned hybrid model's predicted probability of ESI 1-2: (a) reliability curve (plot) and (b) distribution of predicted probabilities.	14
Subgroup and Outcomes	15
Table 4. Under-triage and false alarms by group at a fixed 15% overall miss rate (tuned hybrid without race; n = 83,609 stays with a calculable age)	15
Figure 4. Under-triage of true ESI 1-2 patients by race group and age band for the tuned hybrid model without race, using its default decision rule. (Cells with fewer than 100 urgent patients are blank).	15
Table 5. Hospital admission and ED deaths by triage-error group (test set; admission defined as an ED disposition of admitted)	16
6.	Discussion	16
Interpretation and Feature Importance	16
Figure 5. SHAP summary plot for ESI 1 predictions from the default hybrid model (with race), 2,000 rows	17
Figure 6. SHAP summary plot for ESI 1 predictions from the tuned no-race hybrid model, 2,000 rows.	17
Figure 7. SHAP summary plot for ESI 5 predictions from the default hybrid model (with race), 2,000 rows	18
Figure 8. SHAP summary plot for ESI 5 predictions from the tuned no-race hybrid model, 2,000 rows.	18
Fairness and Equity	19
Clinical Consequences	19
Transparency, Privacy, and Governance	19
Table 6. Principal risks in data preparation and modelling, and what remains	20
Limitations	20
7.	Conclusion	20
References	22
Code and Supporting Material	25
Declaration of Artificial Intelligence (AI) and Supporting Tool Usage	25








ABSTRACT
Background/Purpose: Triage nurses assign each emergency department (ED) patient an acuity level, but the assignment depends on judgment. This project asked whether the Emergency Severity Index (ESI) level recorded at triage can be predicted from chief complaint text and initial vital signs, whether combining the two helps, and whether errors fall evenly across age, sex and race.
Method: Using 418,100 stays from the Medical Information Mart for Intensive Care IV, Emergency Department module (MIMIC-IV-ED), I built text-only (Term Frequency-Inverse Document Frequency, TF-IDF), structured-only (eXtreme Gradient Boosting, XGBoost) and hybrid models. Data were split by patient after a visit-level split placed 33,870 patients in both training and test sets. Class imbalance was handled by weighting, hyperparameters were tuned with patient-grouped cross-validation, and performance was audited by subgroup, hospital outcome and calibration.
Results: On the test set (n = 83,620), the tuned hybrid reached a macro F1 of 0.5527 (95% Confidence Interval [CI] 0.540-0.565), against 0.4020 for the tuned structured model and 0.3656 for text only. Better classification was not safer: the tuned hybrid predicted ESI 3-5 for 30.4% of true ESI 1-2 patients, compared with 22.3% for the default hybrid. At a fixed 15% miss rate, under-triage was highest in 18-34-year-olds (20.0%) and in women (17.0% vs 13.2%), and differed little by race after age adjustment.
Conclusion: Chief complaint text and vital signs carry complementary information, but the model reproduces nurse-assigned labels, and their uneven errors from one US hospital. It is not a replacement for triage; rather, it is a prototype for decision support. 







1.	Background
Triage decides who is seen first in an Emergency Department (ED), and in many hospitals, a nurse makes that decision within minutes. The Emergency Severity Index (ESI) sorts patients into five stages, from 1 (immediate life-saving intervention) to 5 (least urgent), using both the likely seriousness of the presentation and the resources the patient is expected to need (Gilboy et al., 2012). Because the algorithm leaves room for judgement, the label is only as consistent as the people applying it. When nurses in three countries triaged standardised scenarios, accuracy averaged 59.2% and inter-rater reliability was modest (Mistry et al., 2018). Mis-triage has clinical consequences: in a Brazilian ED, critical outcomes and hospital admission rates were greater in under-triaged patients. Over-triaged patients had lower ones (Hinson et al., 2018).
There are various ways that machine learning has been used in triage. Raita et al. (2019) found that models using demographics, vital signs, chief complaint and comorbidities predicted critical care and hospitalisation better than a reference model built on ESI alone. Hong et al. (2018) predicted admission from triage information and prior-visit history, and Fernandes et al. (2020) combined structured triage variables with chief complaint text to predict Intensive Care Unit (ICU) admission. These studies mostly target downstream outcomes. Studies that predict the acuity label itself were less common in what I found; Ivanov et al. (2020) trained a model with clinical natural language processing on 166,175 encounters from two hospitals and tested it against a gold set re-labelled by study clinicians.
Two things stood out to me. In routine data, the only acuity label available is the one the triage nurse assigned, so a model trained on it learns to reproduce nursing practice, inconsistencies included.  In the studies I read, performance was mostly summarised overall, although the choice of label can itself build bias into a health algorithm (Obermeyer et al., 2019). This project therefore asks how well routinely recorded triage information predicts the recorded ESI level, and who is missed when it does not.
The Australasian Triage Scale (ATS) and the ESI both sort patients into five urgency categories, but the two scales are defined differently (Australasian College for Emergency Medicine [ACEM], 2023), so a model trained on ESI labels would need re-validation on local data before any New Zealand use. I therefore treat this work as a methodological prototype.
2.	Objectives
Research question: Among ED stays recorded in Medical Information Mart for Intensive Care IV, Emergency Department module (MIMIC-IV-ED), how well can the chief complaint text and the initial vital signs be used to predict the ESI acuity level allocated at triage, does combining the two data types outperform either alone, and are prediction errors, particularly the under-triage of ESI 1-2 patients, distributed evenly across age, sex and race? The objectives were to:
1.	build a patient-level, leakage-checked training and test dataset from the triage records and document the data quality problems found along the way;
2.	compare text-only, structured-only and hybrid models;
3.	tune and evaluate them with measures that respect class imbalance and clinical risk using macro F1, under-triage, calibration, with confidence intervals;
4.	examine errors by age, sex and race, and by what happened to patients afterwards; and
5.	assess the prototype against fairness, transparency, and privacy and governance principles.
3.	Introduction
Derived from Boston's Beth Israel Deaconess Medical Center, MIMIC-IV-ED contains de-identified records for more than 425,000 emergency department visits spanning 2011 to 2019 (Johnson et al., 2023a) and is distributed to credentialed users via PhysioNet (Goldberger et al., 2000; Pollard et al., 2026). To compute age at each visit, we linked triage and ED stay variables, including vital signs, pain scores, chief complaints, mode of arrival, and nurse-assigned ESI levels, directly to the patient table in the main MIMIC-IV hospital module (Johnson et al., 2023b, 2024). Age was used only to audit the model, never as an input.
The class distribution shapes every design decision. ESI 3 accounts for 53.8% of stays and ESI 5 for 0.3%, so a model that predicts ESI 3 for almost everyone achieves a respectable accuracy while ignoring the patients' triage exists to find. I therefore used macro F1 as the main metric and tracked under-triage, meaning true ESI 1-2 patients predicted as ESI 3-5, as the clinically meaningful error.
The hybrid design combined vital signs and chief complaint. Vital signs record how the patient measures up at the door; the chief complaint records what the nurse was told. Neither is enough on its own in ED practice, and comparing the three arrangements shows what each contributes. 
Three kinds of data are involved: structured measurements, free text and patient demographics. Each fails in its own way. Measurements go missing, free text is unpredictable, and demographics can stand in for things that should not drive triage. Handling each of these properly is most of the work in this report.
This is a prototype built on retrospective data from one US hospital, not a validated clinical tool. Data was accessed through credentialed PhysioNet permissions in a fully de-identified format; I did not attempt to link or re-identify patients, and results are reported in aggregate.

4.	Methods
Data Preparation
Cohort and labels. Linking the ED stay, triage, and patient tables gave 425,087 stays. I removed the 6,987 without a recorded ESI level, leaving 418,100 stays, which became the 334,480-stay training set and the 83,620-stay test set. ESI 3 made up 53.8% of stays, ESI 2 33.3%, ESI 4 6.8%, ESI 1 5.7% and ESI 5 only 0.3%, with only 220 test stays.
Splitting by patient. My first split was a stratified random 80/20 split of visits. Before trusting any score, I counted patients across the two files: 33,870 patients appeared in both, which is 54.8% of the 61,838 patients in the test file. Because frequent attenders generate more visits, these patients accounted for 64.7% of all test visits (roughly 54,000 of 83,620). Repeat visits share complaints, vital signs and history, so evaluating a model on patients it has already seen can flatter its performance (Saeb et al., 2017); it is a little like testing a triage nurse on patients that assessed last month. I switched to a stratified group split based on patient IDs (5-fold, one held out with a random seed of 42). This resulted in 334,480 training stays and 83,620 test stays (20.0%), completely eliminating patient overlap while keeping the exact same ESI proportions in both sets. In practice, the correction changed macro F1 very little (see Results), but it removed a flaw I could not otherwise rule out.
Implausible vital signs. Some entries were clearly keying errors, such as a temperature of 986 or a systolic pressure of 151,103. I first deleted training rows outside plausible ranges (2,148 rows), but the deleted rows contained twice the usual share of ESI 1, which suggested that sick patients were being removed along with the typos. I switched to setting values outside the set to missing: heart rate 30-220, temperature 90-110 °F, systolic 50-220, diastolic 30-140, respiratory rate 8-60, oxygen saturation 50-100, using the same rule in both sets.  This changed 2,268 training and 610 test values and kept every stay.
Missing values. Between 8,186 (heart rate) and 13,251 (temperature) training values were already missing, about 2.4% to 4.0% of the training stays. For each vital sign, I created a missingness indicator before imputing with the training-set median, applied unchanged to the test set. The indicators matter because missingness was not random: a missing temperature or an empty pain score proved to be among the more informative features (see the Results below).
The pain field. Pain was stored as free text. Most entries were scores from 0 to 10 (308,459 training stays), but 9,189 were variants of “unable”, “UTA” or “critical”, 9,017 were numbers outside the scale (mostly 13), 4,964 were empty, and 2,851 were other text. The groups did not look alike: ESI 1 made up 52.1% of empty entries and 38.7% of unable/critical entries, against 3.6% of ordinary scores. Treating every non-numeric entry as missing would have discarded that signal. I kept a 0-10 score where a few spreadsheet-damaged ranges such as “7-Jun”, which appeared to be “6-7”, were restored to midpoints, filled gaps with the training median of 4, and added three indicators for empty, unable/critical, and other or out-of-scale entries.
Quality checks. The missingness indicators flagged between 2.5% (heart rate) and 4.2% (temperature) of stays, with near-identical shares in training and test, which suggests that missingness behaved similarly in the two sets.
Other fields. Chief complaint was missing for 15 training and 2 test stays and was replaced by an empty string. Sex was coded 0/1; applied one-hot encoding to race (33 categories) and mode of arrival ( ambulance, helicopter, walk-in, other, unknown), with test set columns subsequently matched to the training set's. Every step that learns from the data, such as medians, vocabulary, class weights, and encodings, used the training set only.
Feature Selection and Engineering
Inputs were restricted to information available at the moment of triage. Identifiers, timestamps, the hospital admission number, ED departure time and the recorded disposition describe what happens after triage, so they were excluded, and I checked programmatically that none reached the feature matrices. The structured set had 55 features: six vital signs, the pain score, sex, six vital sign missingness indicators, three pain indicators, 33 race categories and five arrival categories. Age was deliberately not a model input.
For text, chief complaints were converted with Term Frequency-Inverse Document Frequency Weighting (TF-IDF; Salton & Buckley, 1988), keeping the 1,000 most frequent terms after removing English stop words, with the vocabulary learned from training text only. The hybrid matrix joined these 1,000 text columns to the structured features (1,055 columns). Because race is a protected characteristic, I tested whether the race columns were needed; removing them left 22 structured features (1,022 hybrid columns), and the comparison is reported below. As an alternative to TF-IDF, I also tried ClinicalBERT embeddings (Alsentzer et al., 2019), via the Hugging Face Transformers library (Wolf et al., 2020), on subsamples of 2,000 and 4,000 test stays.
Model Selection
I compared three pipelines so that the contribution of each data type could be separated: text only, structured only, and the two combined. For text, I used logistic regression on TF-IDF features, a transparent baseline, and compared it with the same classifier on ClinicalBERT embeddings. For structured and hybrid inputs, I selected XGBoost (Chen & Guestrin, 2016), which handles mixed numeric and binary columns and sparse inputs, can use missing values directly, and captures interactions, such as a normal heart rate being reassuring for one complaint but not another. Its weakness here is interpretability, which I addressed with SHAP (see Model evaluation). 
The ClinicalBERT comparison was a check on whether a larger language model was worth the extra cost. On a 4,000-stay subsample, macro F1 was 0.3536 for ClinicalBERT and 0.3392 for TF-IDF; that gap was smaller than the difference between TF-IDF's subsample and full-test scores (0.3392 against 0.3656), so I kept TF-IDF, which was also fast enough to run on every stay. The models assume that triage-time information predicts the nurse's label and that visits from different patients are independent. The second assumption is reasonable after the patient-level split. The first is partly Srcular, because the nurse saw the same vital signs and complaint.
Training and Validation
The test set served solely for the final comparison; all setting selections relied on internal cross-validation (CV) within the training data, with folds again split by patient. 
Class imbalance was handled by weighting training rows by the inverse frequency of their class (He & Garcia, 2009), raised to a power between 0 (no weighting) and 1 (full weighting) that I treated as a tunable setting. I did not oversample, because duplicating or synthesising a class with so few real examples risks memorising them. Hyperparameters (tree count, learning rate, tree depth, minimum child weight, row-column sampling, and an L2 penalty) were searched on an 80,000-row training subsample for the hybrid and a 100,000-row subsample for the structured model, using three patient-grouped folds, twelve random configurations plus the XGBoost defaults, and unweighted macro F1 as the criterion. The best configuration was refitted on full training set and scored a single time on the test set.
My first hyperparameter search went wrong in a way worth recording. I passed the class weights into scikit-learn’s search function, which also applied them when scoring the folds, so the reported cross-validated macro F1 of 0.6054 was a weighted score; re-scored without weights, the same settings gave 0.4507. Weighted scoring rewards over-predicting rare classes (especially ESI 5), which is what the search selected. On the test set, the selected model scored 0.4367, below the default (0.4625), and under-triage rose from 22.3% to 25.2%. I rewrote the search as an explicit loop with unweighted scoring and entered the default settings as a candidate.
Across the thirteen candidates, CV macro F1 ranged from 0.4281 to 0.5288. The top two configurations were effectively tied with 0.5288 and 0.5287, with fold-to-fold standard deviations of 0.0137 and 0.0040, so the choice between them was not decisive; what mattered was that all of the best configurations used softer weights than the default.
Finally, to decide the specific probability cutoff for flagging the urgent patient, the model was trained on 80% of the training patients, and the remaining 20% was used to find the threshold that hit my target miss rate. This cutoff was then locked in and tested on the final test set.
Model Evaluation
Accuracy is dominated by ESI 3, so macro F1 is the main metric, alongside per-class precision and recall to evaluate five classes evenly. The clinically important error is under-triage: a true ESI 1-2 patient predicted ESI 3-5. The reverse, a true ESI 3-5 patient predicted ESI 1-2, is a false alarm. Probability quality was checked with reliability curves, the Brier score and expected calibration error, and the weighted mean difference between predicted and observed risk across distinct probability bins (Van Calster et al., 2019).
Uncertainty was estimated by resampling test patients, not visits, 500 times (Efron & Tibshirani, 1993), giving intervals for each metric and for paired differences between models. Subgroup proportions carry Wilson score intervals (Wilson, 1927), which treat visits as independent and are therefore slightly too narrow. Groups were age bands (18-34, 35-49, 50-64, 65-79, 80+), sex, and five race groups collapsed from the 33 categories. Age came from the anchor age and year in the patient table. Under-triage by race and sex was also age-standardised, weighting each group's age-specific rates by the age mix of all urgent patients. To ask whether errors mattered clinically, I compared admission rates across four groups: urgent patients flagged or missed, and non-urgent patients flagged or not. Admission was defined from the ED disposition, with the presence of a hospital admission number as a sensitivity check.
Explanations used Shapley Additive exPlanations (SHAP) values from TreeExplainer (Lundberg & Lee, 2017; Lundberg et al., 2020) on 2,000 test stays. Analyses were run in Python using scikit-learn (Pedregosa et al., 2011) alongside XGBoost.
5.	Results
Model Comparison
The default hybrid outperformed each single-source model on both macro F1 and under-triage. It reached a macro F1 of 0.4641, against 0.3689 for the unweighted structured model, 0.3656 for text-only and 0.3485 for the class-weighted structured model, and its 95% interval (0.460-0.469) did not overlap any of theirs. Text-only and unweighted structured data were indistinguishable on macro F1 (difference −0.003, 95% CI −0.008 to 0.002) but failed differently: text-only missed 24.7% of urgent patients, the unweighted structured model 53.3%. Class weighting traded accuracy for coverage: for the structured model, it lowered accuracy from 0.616 to 0.392 and ESI 3 recall from 0.85 to 0.26, but lifted ESI 4 and 5 recall from about zero to 0.55 and 0.48 and cut under-triage from 53.3% to 36.5%. The hybrid kept the lowest under-triage (22.3%) with balanced recall (0.80, 0.65, 0.49, 0.74 and 0.50 for ESI 1 to 5), although its precision for ESI 4 and 5 stayed low (0.28 and 0.04). 
Scores before and after correcting the patient overlap differed by less than 0.003 for every model (for the hybrid, 0.4614 before and 0.4641 after), so the overlap did not inflate the main metric here (Table 1; Figure 1).
Model	Macro F1 (95% CI)	Under triage of ESI 1-2 with 95% CI	Main strength	Main weakness	Suitability
Text-only (TF-IDF)	0.3656 (0.362-0.370)	24.7% (24.1-25.3)	Safest, single source; simple; accuracy (0.483)	Weak on rare classes, no physiology	Baseline and hybrid component
Structured, unweighted	0.3689 (0.366-0.372)	53.3% (52.6-54.0)	Second highest accuracy (0.616)	Rarely predicts ESI 4-5; misses half of urgent patients	Not suitable
Structured, class-weighted (default)	0.3485 (0.345-0.352)	36.5% (35.8-37.2)	Finds rare classes	Lowest accuracy (0.392); ESI 3 recall 0.26	Not suitable
Structured, tuned	0.4020 (0.398-0.407)	37.1% (36.4-37.7)	Better balance than defaults with accuracy of 0.522	Still misses over a third of urgent patients	Not suitable alone
Hybrid, default, with race	0.4641 (0.460-0.469)	22.3% (21.6-22.8)	Lowest under triage with accuracy of 0.578	Uses race; low ESI 4-5 precision	Reference only
Hybrid, default, no race	0.4625 (0.458-0.467)	22.3% (21.7-22.9)	Same performance without race	Low ESI 4-5 precision with accuracy of 0.577	Safest default rule option
Hybrid, tuned, no race	0.5527 (0.540-0.565)	30.4% (29.7-31.0)	Best macro F1 and accuracy (0.700); calibrated	Misses more urgent patients at its default rule	Best model for local retraining, with a chosen cutoff
Table 1. Test-set performance of seven models with 83,620 stays and their suitability in New Zealand

 
Figure 1. Default text-only, structured-only and hybrid models: (a) recall by ESI level (plot), (b) macro F1 (bar) and (c) confusion matrix of the default hybrid model (imshow).
Tuning
In the corrected search, the default settings ranked 12th of 13 candidates (Table 2). Tuning raised the hybrid’s test macro F1 to 0.5527 (paired difference +0.0899, 95% CI 0.078-0.102) and the structured model’s to 0.4020 (+0.033, 0.030-0.036). The tuned hybrid then led the tuned structured model by 0.1502 (0.138-0.163) and missed fewer urgent patients (30.4% against 37.1%). Against the default hybrid, however, it missed more (30.4% against 22.3%, a difference of 8.0 points, 7.7-8.5). The per-class recall shows this: ESI 3 rose from 0.4892 to 0.7949, while ESI 1, 4, and 5 fell (0.796 to 0.6784, 0.7318 to 0.3813, and 0.4773 to 0.1909), so the tuned model became more reluctant to flag anything as unusual.
Model 	Default settings (CV macro F1)	Best configuration found (CV macro F1)	Test macro F1, default to tuned
Hybrid, no race (80,000 rows)	0.4838, ranked 12th of 13th 	0.5288, class-weight power 0.25, depth 6, learning rate 0.3, 200 trees, minimum child weight 10, subsample 0.85, L 2 penalty 5	0.4625 to 0.5527
Structured (100,000 rows)	0.3678 unweighted; 0.3620 class-weighted	0.4018, class-weight power 0.75, depth 4, learning rate 0.3, 200 trees, minimum child weight 1, subsample 0.85, L2 penalty 1	0.3689 to 0.402
Hybrid, first search (weighted scoring error)		0.605 when scored with class weights (0.451 unweighted)	0.437
Table 2. Hyperparameter search (unweighted macro F1, patient-grouped three-fold cross-validation)

Operating point
While holding both models to the same 15% miss rate, the tuned hybrid raised about three fewer false alarms per 100 non-urgent patients at every target (Table 3; 29.3% against 32.9% at a 15% miss rate) and separates urgent from non-urgent stays slightly better (Figure 2; AUC 0.870 vs 0.855). A 15% target is still costly because 39.1% of all stays are urgent; it means flagging about half of all patients (from calculation Table 3)
Target Miss rate	Default hybrid: achieved miss/ false alarms	Tuned hybrid: achieved miss/ false alarms
25%	24.8% / 21.2%	 25.3% / 18.0%
20%	19.9% / 26.2%	20.2% / 22.8%
15%	14.7% / 32.9%	15.1% / 29.3%
10%	9.8% / 42.3%	9.8% / 39.2%
Table 3. Test-set miss rate and false-alarm rate when each hybrid model is set to miss a chosen share of urgent patients.

 
Figure 2. Trade-off curves between missed urgent patients (under-triage) and false alarms for the default and tuned hybrid models, without race. Dots mark each model's own decision rule.
Calibration
Predicted probabilities of ESI 1-2 from the tuned hybrid tracked observed rates closely (Brier score 0.141, expected calibration error 0.020), with mild over-prediction of urgency (mean predicted 0.408 against 0.391 observed; Figure 3).
 
Figure 3. Calibration of the tuned hybrid model's predicted probability of ESI 1-2: (a) reliability curve (plot) and (b) distribution of predicted probabilities.

Subgroup and Outcomes
Group	Urgent patients (n)	Under-triage (95% CI)	Age- standardised under-triage	False alarms
Sex				
Female	16,554	17.0% (16.4-17.5)	17.1%	27.7%
Male	16,128	13.2% (12.6-13.7)	13.1%	31.4%
Race group 				
Asian/Pacific Islander	1,291	16.0% (14.1-18.1)	15.3%	24.1%
Black/African American	6,017	16.3% (15.4-17.2)	15.7%	27.6%
Hispanic/Latino	1,968	18.3% (16.6-20.1)	16.8%	24.4%
Other/Unknown	2,400	14.5% (13.2-16.0)	13.5%	27.3%
White	21,006	14.4% (14.0-14.9)	14.8%	31.7%
Age band				
18-34	5,824	20.0% (18.9-21.0)	20.0%	23.0%
35-49	5,344	18.5% (17.5-19.6)	18.5%	24.7%
50-64	8,671	16.1% (15.4-16.9)	16.1%	29.3%
65-79	7,690	12.0% (11.3-12.7)	12.0%	38.5%
80+	5,153	8.9% (8.1-9.7)	8.9%	49.2%
Table 4. Under-triage and false alarms by group at a fixed 15% overall miss rate (tuned hybrid without race; n = 83,609 stays with a calculable age)

 
Figure 4. Under-triage of true ESI 1-2 patients by race group and age band for the tuned hybrid model without race, using its default decision rule. (Cells with fewer than 100 urgent patients are blank).

Group (tuned hybrid, default decision rule)	Stays	Admitted from ED (95% CI)	Deaths in ED
A. Urgent (ESI 1-2), missed (predicted ESI 3-5)	9,928	44.7% (43.7-45.6)	2
B. Urgent (ESI 1-2), flagged (predicted ESI 1-2)	22,758	61.8% (61.2-62.4)	25
C. Non-urgent (ESI 3-5), flagged (predicted ESI 1-2)	7,220	40.6% (39.4-41.7)	2
D. Non-urgent (ESI 3-5), not flagged (predicted ESI 3-5)	43,714	21.2% (20.8-21.6)	3
Table 5. Hospital admission and ED deaths by triage-error group (test set; admission defined as an ED disposition of admitted)
6.	Discussion
Interpretation and Feature Importance
Text and vital signs carry fundamentally different clinical information. The text-only model proved to be the safest single data source, missing 24.7% of urgent patients compared to 37.1% for the tuned structured model, which fits clinical intuition: a complaint such as chest pain or numbness raises concern before any measurement does. 
Comparing SHAP values reveals consistent behaviour in identifying high-acuity patients. For ESI 1 predictions, both the default hybrid (Figure 5) and the tuned no-race hybrid (Figure 6) rely on nearly identical top features. In both models, a high heart rate, low systolic pressure, low oxygen saturation, and high respiratory rate push predictions towards ESI 1. Arrival by ambulance pushes the prediction in the same direction, while walking in pulls it the other way. Less comfortably, both models rely heavily on missing data: a missing temperature, a missing respiratory rate, and an empty pain score strongly push the model toward an ESI 1 prediction. The patients requiring immediate, life-saving resuscitation are assessed before a full set of observations, and a pain score is recorded. Under this scenario, the model's predictive performance would likely degrade if clinical documentation practices changed.
For non-urgent ESI 5 predictions, the models behave with subtle differences. Both the default (Figure 7) and tuned no-race (Figure 8) models agree that arriving by ambulance or recording high pain strongly pulls the prediction away from ESI 5. However, the default model actively uses administrative text like "refill" and "removal" to push towards an ESI 5 classification. In contrast, these administrative terms drop out of the top 20 features in the tuned no-race model, which instead focuses on clinical complaints such as "chest" to push predictions away from ESI 5.
 
Figure 5. SHAP summary plot for ESI 1 predictions from the default hybrid model (with race), 2,000 rows

 
Figure 6. SHAP summary plot for ESI 1 predictions from the tuned no-race hybrid model, 2,000 rows.

 
Figure 7. SHAP summary plot for ESI 5 predictions from the default hybrid model (with race), 2,000 rows
 
Figure 8. SHAP summary plot for ESI 5 predictions from the tuned no-race hybrid model, 2,000 rows.

Fairness and Equity
Errors were not evenly distributed, and some gaps were more stubborn than others. Age showed the largest disparity: at a fixed 15% miss rate, 20.0% of urgent patients aged 18-34 were missed, against 8.9% of those aged 80 and over (Table 4). Age was not an input, so the model is probably reading correlates such as vital sign patterns or arrival mode. Women were missed more often than men (17.0% against 13.2%), and the gap remained after age standardisation, suggesting it is not simply an age effect. Race behaved differently than under the default decision rule: the Black-White under-triage gap was 6.2 percentage points and the Hispanic-White gap 8.3; removing the race columns cut these to 2.4 and 5.4 without lowering macro F1, which supports excluding race as an input. With a fixed miss rate and age standardisation, the gaps shrank further (Black 0.9 and Hispanic/Latino 2.0 points above White patients), although tuning raised under-triage in every group (by 5.9 to 9.2 points) and left the default-rule gaps in place. Two cautions apply: all results are measured against nurse-assigned labels, and a model trained on them can copy patterns in nursing triage (Hinson et al., 2018), so a gap against the label is not necessarily a gap against clinical need (Obermeyer et al., 2019; Rajkomar et al., 2018). Groups also differ in complaint mix, arrival mode and documentation, and some cells are small (for example, 140 urgent Hispanic/Latino patients aged 80+; Figure 4), so these are patterns to investigate, not causes.
Clinical Consequences
The tuned model’s misses were not random: urgent patients it missed were admitted less often than those it flagged (44.7% against 61.8%; Table 5), so they were probably, on average, less sick. Even so, 4,434 admitted patients (5.3% of the test set) were predicted as non-urgent.
Transparency, Privacy, and Governance
Incorporating TF-IDF terms and SHAP values offer visibility into model behaviour, though SHAP reflects algorithmic decision-making rather than underlying patient physiology. Calibration supports presenting clinicians with probabilities rather than hard labels (Van Calster et al., 2019). Ethical data standards were met by using credentialed, de-identified data and sharing only aggregated results. This framework reflects global and national guidelines, including World Health Organisation (WHO, 2021) standards for transparent, inclusive, and accountable AI, New Zealand's Algorithm Charter requirements for Treaty partnership and human governance (Stats NZ, 2020), and Māori data sovereignty expectations regarding disaggregated health data (Te Mana Raraunga, 2018).MIMIC-IV’s race categories contain no Māori group, so this work cannot assess equity for Māori patients, the largest gap for any New Zealand use. Deployment would add further risks (Table 6): clinicians may over-trust a score, workflow-dependent features may drift, and someone must be accountable for choosing and monitoring the cutoff.

Risk	What I did	What Remains
Patient overlap between training and test	Stratified patient-level split; confirmed zero shared patients	Repeat visits still make interval estimates slightly narrow
Implausible vital signs	Set to missing in both sets, with indicators; no stays deleted	Genuine extreme readings, such as in shock, are treated as missing
Missing data	Training-median imputation plus missingness indicators	Missingness reflects local documentation habits, not physiology
Free-text pain scores	Parsed to a 0-10 score and three indicators	“No pain score” predicts ESI 1 through workflow and could change
Class imbalance	Weights with tuned strength; macro F1; per-class recall	ESI 5 has only 220 test stays, so its estimates are unstable
Label noise and bias	Reported against nurse labels; checked admission outcomes	Model error cannot be separated from label error
Protected attributes	Race removed after ablation; age used only for auditing	Sex retained; age and race can still be inferred from correlates
Small subgroups and confounding	Wilson intervals; age standardisation; cells under 100 blank in heatmap	Complaint mix, arrival mode and clustered visits are not adjusted for
Table 6. Principal risks in data preparation and modelling, and what remains
Limitations
The data come from a single US centre during one period, and all labels are nurse-assigned rather than outcome-verified. Only chief complaint text was available, not detailed nursing notes. ESI 5 estimates are unstable with only 220 test cases. There is no temporal or external validation, and the fairness analysis is purely associational, so it cannot explain why groups differ.
7.	Conclusion
On this dataset, chief complaint text and initial vital signs together predicted ESI acuity better than either alone. The tuned hybrid reached a macro F1 of 0.5527 (95% CI 0.540-0.565) against 0.4020 for tuned structured data and 0.3656 for text, and removing race as an input did not reduce performance. That does not make it a triage tool. Under its default decision rule, it predicted ESI 3-5 for 30.4% of true ESI 1-2 patients; errors were largest in young adults and women, and every result is relative to nurse-assigned labels from one hospital in the United States.
None of the seven models is ready for use in a New Zealand ED. The labels are ESI levels from one Boston hospital, whereas New Zealand departments work with the Australasian Triage Scale (ACEM, 2023); MIMIC’s race categories contain no Māori group, so equity for Māori patients cannot be assessed; and the Algorithm Charter (Stats NZ, 2020) and Māori data sovereignty principles (Te Mana Raraunga, 2018) expect partnership, transparency and human oversight that a prototype trained overseas does not meet by default. What transfers is the method: patient-level splitting, scoring tuning with the reported metric, choosing the operating point on validation data and auditing by group. If a New Zealand ED pursued this, the sensible candidate would be the tuned hybrid without race, retrained on local ATS-labelled data, run silently alongside nurses with a cutoff set to hold under-triage at a level clinicians accept, audited by ethnicity including Māori, and monitored over time (WHO, 2021).















References
Alsentzer, E., Murphy, J., Boag, W., Weng, W.-H., Jin, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. In Proceedings of the 2nd Clinical Natural Language Processing Workshop (pp. 72-78). Association for Computational Linguistics. https://doi.org/10.18653/v1/W19-1909
Australasian College for Emergency Medicine. (2023, November). Policy on the Australasian Triage Scale (Version 5, P06). https://acem.org.au/getmedia/484b39f1-7c99-427b-b46e-005b0cd6ac64/P06_Policy_Australasian_Triage_Scale
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785
Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.
Fernandes, M., Mendes, R., Vieira, S. M., Leite, F., Palos, C., Johnson, A., Finkelstein, S., Horng, S., & Celi, L. A. (2020). Predicting intensive care unit admission among patients presenting to the emergency department using machine learning and natural language processing. PLOS ONE, 15(3), Article e0229331. https://doi.org/10.1371/journal.pone.0229331
Gilboy, N., Tanabe, P., Travers, D., & Rosenau, A. M. (2012). Emergency Severity Index (ESI): A triage tool for emergency department care, version 4. Implementation handbook, 2012 edition (AHRQ Publication No. 12-0014). Agency for Healthcare Research and Quality. https://www.govinfo.gov/content/pkg/GOVPUB-HE20_6500-PURL-gpo23161/pdf/GOVPUB-HE20_6500-PURL-gpo23161.pdf
Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220. https://doi.org/10.1161/01.CIR.101.23.e215
He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284. https://doi: 10.1109/TKDE.2008.239
Hinson, J. S., Martinez, D. A., Schmitz, P. S. K., Toerper, M., Radu, D., Scheulen, J., Stewart de Ramirez, S. A., & Levin, S. (2018). Accuracy of emergency department triage using the Emergency Severity Index and independent predictors of under-triage and over-triage in Brazil: A retrospective cohort analysis. International Journal of Emergency Medicine, 11, Article 3. https://doi.org/10.1186/s12245-017-0161-8
Hong, W. S., Haimovich, A. D., & Taylor, R. A. (2018). Predicting hospital admission at emergency department triage using machine learning. PLOS ONE, 13(7), Article e0201016. https://doi.org/10.1371/journal.pone.0201016
Ivanov, O., Wolf, L., Brecher, D., Masek, K., Lewis, E., Liu, S., Dunne, R. B., Klauer, K., Montgomery, K., Andrieiev, Y., McLaughlin, M., & Reilly, C. (2020). Improving emergency department ESI acuity assignment using machine learning and clinical natural language processing [Preprint]. arXiv. https://arxiv.org/abs/2004.05184
Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023a). MIMIC-IV-ED (Version 2.2) [Data set]. PhysioNet. https://doi.org/10.13026/5ntk-km72
Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (Version 3.1) [Data set]. PhysioNet. https://doi.org/10.13026/kpb9-mt58
Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L. H., Celi, L. A., & Mark, R. G. (2023b). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10, Article 1. https://doi.org/10.1038/s41597-022-01899-x
Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56-67. https://doi.org/10.1038/s42256-019-0138-9
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. Guyon, U. von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in neural information processing systems 30 (pp. 4765-4774). Curran Associates.https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
Mistry, B., Stewart De Ramirez, S., Kelen, G., Schmitz, P. S. K., Balhara, K. S., Levin, S., Martinez, D., Psoter, K., Anton, X., & Hinson, J. S. (2018). Accuracy and reliability of emergency department triage using the Emergency Severity Index: An international multicenter assessment. Annals of Emergency Medicine, 71(5), 581-587. https://doi.org/10.1016/j.annemergmed.2017.09.036
Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464), 447-453. https://doi.org/10.1126/science.aax2342
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z
Raita, Y., Goto, T., Faridi, M. K., Brown, D. F. M., Camargo, C. A., Jr., & Hasegawa, K. (2019). Emergency department triage prediction of clinical outcomes using machine learning models. Critical Care, 23, Article 64. https://doi.org/10.1186/s13054-019-2351-7
Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018). Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine, 169(12), 866-872. https://doi.org/10.7326/M18-1990
Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. GigaScience, 6(5), Article gix019. https://doi.org/10.1093/gigascience/gix019
Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513-523. https://doi.org/10.1016/0306-4573(88)90021-0
Stats NZ. (2020, July). Algorithm charter for Aotearoa New Zealand. https://www.data.govt.nz/assets/data-ethics/algorithm/Algorithm-Charter-2020_Final-English-1.pdf
Te Mana Raraunga. (2018). Principles of Māori data sovereignty (Brief #1). https://static1.squarespace.com/static/58e9b10f9de4bb8d1fb5ebbc/t/5bda208b4ae237cd89ee16e9/1541021836126/TMR+Ma%CC%84ori+Data+Sovereignty+Principles+Oct+2018.pdf
Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. BMC Medicine, 17, Article 230. https://doi.org/10.1186/s12916-019-1466-7
Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158), 209-212. https://doi.org/10.1080/01621459.1927.10502953
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., . . . Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 38-45). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-demos.6
World Health Organization. (2021). Ethics and governance of artificial intelligence for health: WHO guidance. https://www.who.int/publications/i/item/9789240029200

Code and Supporting Material
The scripts used for data preparation, model training, tuning, and evaluating the models are submitted alongside this report as Jupyter notebooks, in the order they were run:
•	01_data_prep: cohort construction, patient-level split, vital sign and pain cleaning, missingness indicators, imputation and encoding;
•	Text_Model_A: TF-IDF and ClinicalBERT text-only models, with the three-panel figure for the text models;
•	Structured_Model_B: unweighted and class-weighted XGBoost, the corrected hyperparameter search and the structured-model figure;
•	Hybrid_Model_C: hybrid model, race ablation, corrected search, operating points, calibration, subgroup and outcome analyses, SHAP;
MIMIC datasets cannot be redistributed under the PhysioNet data use agreement, so raw data are not included; access requires credentialed registration at PhysioNet. Because foundational demographics (patients.csv) and emergency department variables span separate MIMIC-IV (v3.1) and MIMIC-IV-ED (v2.2) modules, both are cited independently due to distinct DOIs. No credentials or access tokens are stored in the submitted code. Random seeds were fixed (42) for the data split and model training. Figures were generated using programmatic code scripts.
Declaration of Artificial Intelligence (AI) and Supporting Tool Usage
I used generative AI and digital tools during the project, as follows:
•	Claude: writing, troubleshooting and refining the Python code in all notebooks; planning and explaining the numerical results; drafting and revising the report text and counting words; matching text author citations against my final reference list.
•	Gemini: cross-checking the Python code, drafting the report
•	Visual Studio Code Insiders with GitHub Copilot: Explaining the code and error correction. 
•	Grammarly: Correcting readability and grammar
•	Research and External Sources: PhysioNet and MIMIC documentation and literature found through Google and journal websites.
I ran all code locally, and all outputs in this report originate directly from my notebooks. While AI tools assisted in generating text and code, I personally reviewed, edited, and verified everything. I accept complete accountability for the academic integrity, referencing, and final quality of this submission.
