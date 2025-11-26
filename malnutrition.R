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


data_malnutrition<-read_excel(file.path(results_folder,"Malnutrition recode - alibayram__medgemma:27b.xlsx"))



df = left_join(data_malnutrition %>% select(record_id,whz,doc_muc,malnutrition_diagnosis, malnutrition_risk,nutritionist_referred),
outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("record_id"="record_id")) %>% 
filter(n_clinicians==1)


df <- df  %>% 
mutate(malnutrition=case_when(
  doc_muc %in% c("Yellow", "Red")~1.0,
  whz <= -2 ~1.0,
  TRUE ~0.0
),identified=case_when(
  malnutrition==0~NA_real_,
  malnutrition_diagnosis ==T | nutritionist_referred ==T~1.0,
  malnutrition_diagnosis == F~0.0,
),
treated=case_when(
  malnutrition==0~NA_real_,
  nutritionist_referred ==T~1.0,
  nutritionist_referred == F~0.0,
))

tab_referred<- df  %>% filter(malnutrition==1) %>% mutate(referred=if_else(nutritionist_referred == T, "Yes", "No")) %>%
  tbl_summary(include=arm,by=c(referred),
  label=list(arm="Study Arm"),
  missing="no",percent = "row")

model_referred <- glmer(nutritionist_referred ~ arm + (1 | hospital_num) + (1 | clinician_num), data = df %>% filter(malnutrition==1), family = binomial)

summary(model)
reg_referred<-tbl_regression(model_referred, exponentiate = TRUE,
label=list(arm="Study Arm")) 
tbl_merge(list(tab_referred,reg_referred),tab_spanner = c("**Referred to Nutritionist**", "**Multilevel logistic regression**"))%>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "malnutrition_referred_model_results.html")  )



tab_identified<- df  %>% filter(malnutrition==1) %>% mutate(identified=if_else(identified == T, "Yes", "No")) %>%
  tbl_summary(include=arm,by=c(identified),
  label=list(arm="Study Arm"),
  missing="no",percent = "row")

model_identified <- glmer(identified ~ arm + (1 | hospital_num) + (1 | clinician_num), data = df %>% filter(malnutrition==1), family = binomial)

reg_identified<-tbl_regression(model_identified, exponentiate = TRUE,
label=list(arm="Study Arm")) 
tbl_merge(list(tab_identified,reg_identified),tab_spanner = c("**Diagnosed with malnutrition**", "**Multilevel logistic regression**"))%>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "malnutrition_identified_model_results.html")  )



sankey_data<-df %>% 
mutate(malnutrition=case_when(
  malnutrition==1~"Has malnutrition",
  malnutrition==0~"No malnutrition"
),
identified=case_when(
  identified==1~"Diagnosed",
  identified==0~"Not diagnosed",
  TRUE ~NA_character_
),
treated=case_when(
  treated==1~"Referred to nutritionist",
  treated==0~"Not referred to nutritionist",
  TRUE ~NA_character_
))

#sankey plot

