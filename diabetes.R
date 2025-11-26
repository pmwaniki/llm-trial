library(arrow)
library(gtsummary)
library(gt)
library(lme4)
library(stringr)
library(janitor)
library(performance)
library(ggplot2)
library(ggthemes)
library(tidyverse)
library(readxl)





# load environment variables from .env file

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")

outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))
# metrics=with(data,anthro_zscores(sex=sex,age=age_mths,is_age_in_month = T,weight = weight_kgs,lenhei = height_cm))


data_diabetes<-read_excel(file.path(results_folder,"Diabetes recode - alibayram__medgemma:27b.xlsx"))



df = left_join(data_diabetes %>% select(record_id,diabetes_at_risk, diabetes_test, diabetes_positive, diabetes_treated,diabetes_chronic),
outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("record_id"="record_id")) %>% 
filter(n_clinicians==1)



df <- df %>% 
mutate(risk=case_when(
  diabetes_chronic==T ~NA_real_,
  diabetes_at_risk ==T ~1.0,
  diabetes_at_risk == F~0.0,
),
tested=case_when(
  risk %in% c(0,NA) | diabetes_chronic==T ~NA_real_,
  diabetes_test ==T ~1.0,
  diabetes_test == F~0.0,
),positive=case_when(
  !(tested %in% c(1))~NA_real_,
  diabetes_positive ==T ~1.0,
  diabetes_positive == F~0.0,
),treated=case_when(
  !(positive %in% c(1))~NA_real_,
  diabetes_treated ==T ~1.0,
  diabetes_treated == F~0.0,
))


diabetes_tab<-tbl_summary(df,
include=c(diabetes_chronic,risk,tested,positive,treated),
by=arm,
label=list(
  diabetes_chronic="Chronic Diabetes Diagnosis",
  risk="At Risk of Diabetes",
  tested="Tested for Diabetes",
  positive="Diagnosed with Diabetes",
  treated="Treated for Diabetes"
),
missing="no",percent = "column") %>% 
add_overall() %>% 
modify_table_styling(
    columns = label,
    rows = label == "At Risk of Diabetes",
    footnote = "Excludes patients with chronic diabetes"
  )%>% 
modify_table_styling(
    columns = label,
    rows = label == "Tested for Diabetes",
    footnote = "Includes only patients at risk of diabetes"
  )  %>% 
modify_table_styling(
    columns = label,
    rows = label == "Diagnosed with Diabetes",
    footnote = "As a fraction of those tested for diabetes"
  )  %>% 
modify_table_styling(
    columns = label,
    rows = label == "Treated for Diabetes",
    footnote = "As a fraction of those diagnosed with diabetes"
  )

diabetes_tab %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "diabetes_summary.html"))


tab_risk<- df  %>% mutate(risk=if_else(risk == 1, "Yes", "No")) %>%
  tbl_summary(include=arm,by=c(risk),
  label=list(arm="Study Arm"),
  missing="no",percent = "row")
model_risk <- glmer(risk ~ arm + (1 | hospital_num) + (1 | clinician_num), data = df, family = binomial)
reg_risk<-tbl_regression(model_risk, exponentiate = TRUE,
label=list(arm="Study Arm"))
final_risk<-tbl_merge(list(tab_risk,reg_risk),tab_spanner = c("**Summaries**", "**Logistic regression**"))


tab_tested<- df  %>% filter(risk==1) %>% mutate(tested=if_else(tested == T, "Yes", "No")) %>%
  tbl_summary(include=arm,by=c(tested),
  label=list(arm="Study Arm"),
  missing="no",percent = "row")
model_tested <- glmer(tested ~ arm + (1 | hospital_num) + (1 | clinician_num), data = df %>% filter(risk==1), family = binomial)
reg_tested<-tbl_regression(model_tested, exponentiate = TRUE,
label=list(arm="Study Arm"))

final_tested<-tbl_merge(list(tab_tested,reg_tested),tab_spanner = c("**Summaries**", "**Logistic regression**"))




tab_treated<- df  %>% filter(positive==1) %>% mutate(treated=if_else(treated == T, "Yes", "No")) %>%
  tbl_summary(include=arm,by=c(treated),
  label=list(arm="Study Arm"),
  missing="no",percent = "row")
model_treated<- glmer(treated ~ arm + (1 | hospital_num) + (1 | clinician_num), data = df %>% filter(positive==1), family = binomial)
reg_treated<-tbl_regression(model_treated, exponentiate = TRUE,
label=list(arm="Study Arm"))

model_treated2<- glm(treated ~ arm , data = df %>% filter(positive==1), family = binomial)
reg_treated2<-tbl_regression(model_treated2, exponentiate = TRUE,
label=list(arm="Study Arm"))

final_treated<-tbl_merge(list(tab_treated,reg_treated2),tab_spanner = c("**Summaries**", "**Logistic regression**"))



diabetes_table<-tbl_stack(list(final_risk,final_tested,final_treated),
  group_header = c(
  "At Risk of Diabetes (Multilevel logistic regression Results)",
  "Tested for Diabetes (Multilevel logistic regression Results)",
  "Treated for Diabetes (Logistic regression Results)")) 


diabetes_table%>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "diabetes_model_results.html")  )