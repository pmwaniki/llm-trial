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


data_antimalarials<-read_excel(file.path(results_folder,"Anti-malarial recode - alibayram__medgemma:27b.xlsx"))



df = left_join(data_antimalarials %>% select(record_id,malaria_test,malaria_positive,antimalarial_prescribed),
outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("record_id"="record_id")) %>% 
filter(n_clinicians==1)


df <- df %>%
mutate(has_antimalarial=case_when(
  antimalarial_prescribed==T ~1.0,
  antimalarial_prescribed==F ~0.0,
),
tested=case_when(
    has_antimalarial!=1 ~NA_real_,
    malaria_test ==T ~1.0,
    malaria_test == F~0.0,
),y=case_when(
  has_antimalarial!=1 ~NA_real_,
  tested==0 ~1.0,
  malaria_positive==F ~ 1.0,
  malaria_positive == T~0.0,
))

tab_antimalarial<-tbl_summary(df,
include=c(has_antimalarial,tested,y),
by=arm,
label=list(
  has_antimalarial="Antimalarial Prescribed",
  tested="Malaria Test Performed",
  y="Malaria Test Negative or not Performed"
),
missing="no") %>% add_overall() %>% 
modify_table_styling(
    columns = label,
    rows = label %in% c("Malaria Test Performed","Malaria Test Negative or not Performed"),
    footnote = "Only for patients prescribed antimalarials"
  )

tab_incorrect_antimalarial<- tbl_summary(df %>% filter(has_antimalarial==1) %>% mutate(y=factor(y,levels=c(0,1),labels=c("No","Yes"))), ,
include = c(arm),
by=y,
percent = "row",
label=list(
  arm="Study Arm"
),
missing="no")


model<-glmer(y ~ arm + (1|hospital_num) + (1|clinician_num),data=df %>% filter(!is.na(y)),
          family=binomial(link="logit"),
          control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5))
)
reg<-tbl_regression(model,exponentiate = T,label = list(
  arm="Study Arm"
))

final_incorrect_antimalarial<-tbl_merge(
  tbls = list(tab_incorrect_antimalarial,reg),
  tab_spanner = c("**Incorrect Antimalarial Prescribing**", "**Multilevel logistic regression**")
)

final_incorrect_antimalarial %>% 
as_gt() %>%
gt::gtsave(filename = file.path(output_folder,"antimalarial_incorrect_prescribing_model.html"))

tab_antimalarial %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "antimalarial_summary.html")  )