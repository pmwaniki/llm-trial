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


data_hypertension<-read_excel(file.path(results_folder,"Hypertension recode - alibayram__medgemma:27b.xlsx"))



df = left_join(data_hypertension %>% select(record_id,hypertension_new,hypertension_chronic,hypertension_treated),
outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("record_id"="record_id")) %>% 
filter(n_clinicians==1)

df <- df %>% 
mutate(
    new_htn=case_when(
      hypertension_chronic==T ~NA_real_,
      hypertension_new ==T ~1.0,
      hypertension_new == F~0.0,),
treated=case_when(
  new_htn!=1 ~NA_real_,
  hypertension_treated ==T ~1.0,
  hypertension_treated == F~0.0,
))

tab_chronic_htn<- tbl_summary(df %>%  mutate(hypertension_chronic=factor(hypertension_chronic,levels=c(0,1),labels=c("No","Yes"))),
include = c(arm),
by=hypertension_chronic,
percent = "row",
label=list(
  arm="Study Arm"
),
missing="no")


tab_htn_new<-tbl_summary(df %>% mutate(hypertension_new=factor(hypertension_new,levels=c(0,1),labels=c("No","Yes"))),
include=c(arm),
by=hypertension_new,
label=list(
  arm="Study Arm"
),
missing="no")

model_htn_new<-glmer(new_htn ~ arm + (1|hospital_num) + (1|clinician_num),data=df,family=binomial(link="logit"))
reg_htn_new<-tbl_regression(model_htn_new,exponentiate = T,label = list(
  arm="Study Arm"
))

final_new_htn<-tbl_merge(
  tbls = list(tab_htn_new,reg_htn_new),
  tab_spanner = c("**Summary**", "**Multilevel logistic regression**")
)

tab_treated<- tbl_summary(df %>% filter(new_htn==1) %>% mutate(hypertension_treated=factor(hypertension_treated,levels=c(0,1),labels=c("No","Yes"))),
include=c(arm),
by=hypertension_treated,
label=list(
  arm="Study Arm"
),
missing="no")
model_treated<-glmer(treated ~ arm + (1|hospital_num) + (1|clinician_num),data=df %>% filter(new_htn==1),family=binomial(link="logit"))
reg_treated<-tbl_regression(model_treated,exponentiate = T,label = list(
  arm="Study Arm"
))

final_treated<-tbl_merge(
  tbls = list(tab_treated,reg_treated),
  tab_spanner = c("**Summary**", "**Multilevel logistic regression**")
)  

final_htn<-tbl_stack(
  tbls = list(final_new_htn,final_treated),
  group_header = c("**New Hypertension Diagnosis**","**Treatment for New Hypertension**")
)


final_htn %>% as_gt() %>% gt::gtsave(filename = file.path(output_folder,"hypertension_summary.html"))
