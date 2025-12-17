library(arrow)
library(dplyr)
library(gtsummary)
library(gt)
library(lme4)
library(stringr)
library(janitor)
library(performance)
library(brms)
library(tidybayes)
library(ggplot2)
library(ggthemes)
library(ggridges)
library(glue)
library(readxl)
library(tidyr)




# load environment variables from .env file

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")

outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))

data_malnutrition<-read_excel(file.path(results_folder,"Malnutrition recode - alibayram__medgemma:27b.xlsx"))
antibiotic_data<-read_parquet(file.path(data_folder,"FEVER_COLLATED.parquet")) 

data_hypertension<-read_excel(file.path(results_folder,"Hypertension recode - alibayram__medgemma:27b.xlsx")) %>% 
filter(hypertension_chronic==T | hypertension_new==T)

data_diabetes<-read_excel(file.path(results_folder,"Diabetes recode - alibayram__medgemma:27b.xlsx")) %>% 
filter(diabetes_chronic==T | diabetes_positive==T)

data_antimalarials<-read_excel(file.path(results_folder,"Anti-malarial recode - alibayram__medgemma:27b.xlsx")) %>% 
filter(malaria_positive==T)

operational_data=read_parquet(file.path(data_folder, 'Operational_data.parquet'))


# merge outcome_data with operational data to get registration date
outcome_data<-outcome_data %>%
left_join(operational_data %>% select(`record_id`,`registration`),by="record_id")


# recode age children vs adult
outcome_data<-outcome_data  %>% 
mutate(age_adult=if_else(age_category %in% c("18 to 55 years","55+ years"),"Adult","Pediatric"),
age_adult=factor(age_adult,levels=c("Pediatric","Adult"),ordered=FALSE),
age_category2=factor(age_category,ordered=F))

# recode day weekday vs weekend
outcome_data<-outcome_data %>%
mutate(weekday=format(registration,"%u"),
day_type=if_else(weekday %in% c(6,7),"Weekend","Weekday"),
day_type=factor(day_type,levels=c("Weekend","Weekday"),ordered=FALSE))


#recode day vs night using registration
outcome_data<-outcome_data %>%
mutate(hour_registration=as.numeric(format(registration,"%H")),
time_of_day=if_else(hour_registration>=7 & hour_registration<19,"Day","Night"),
time_of_day=factor(time_of_day,levels=c("Night","Day"),ordered=FALSE))


outcome_data<-outcome_data %>% 
mutate(age_category=factor(age_category,ordered=F),
failure=if_else(failure_num==1,"Failure","No failure"))


data_malnutrition<-data_malnutrition %>% 
mutate(malnutrition=case_when(
  doc_muc %in% c("Yellow", "Red")~1.0,
  whz <= -2 ~1.0,
  TRUE ~0.0
)) %>% filter(malnutrition %in% c(1))

outcome_data <- outcome_data %>% 
mutate(malnutrition=case_when(
  record_id %in% data_malnutrition$record_id ~ 1,
  TRUE ~ 0.
),fever=case_when(
  record_id %in% antibiotic_data$study_id ~ 1,
  TRUE ~ 0.
),
hypertension=case_when(
  record_id %in% data_hypertension$record_id ~ 1,
  TRUE ~ 0.
),diabetes=case_when(
  record_id %in% data_diabetes$record_id ~ 1,
  TRUE ~ 0.
),malaria=case_when(
  record_id %in% data_antimalarials$record_id ~ 1,
  TRUE ~ 0.
),
sentinel_condition=case_when(
  malnutrition==1 | fever==1 | hypertension==1 | diabetes==1 | malaria==1 ~"Yes",
  TRUE ~"No"
)
)


#################################################################################################################################################3
#
# SUBGROUP ANALYSES
#

###################################################################################
## Subgroup analyses

calculate_or_ci <- function(model, base_term,subgroup_name) {
  subgroup_levels=unique(model@frame[[subgroup_name]])
    coefs<-fixef(model)
    vcov_matrix<-vcov(model)
    result_levels=list()
    for(level in subgroup_levels){
      interaction_term=paste0(base_term,":",subgroup_name,level)
      if(!interaction_term %in% names(coefs)){
        interaction_term=NA
      }
      base_coef <- coefs[[base_term]]
    var_base <- vcov_matrix[base_term, base_term]
    
    if (!is.na(interaction_term)) {
      if (!interaction_term %in% names(coefs)) {
        stop(paste("Interaction term not found in coefficients:", interaction_term))
      }
      interaction_coef <- coefs[[interaction_term]]
      var_interaction <- vcov_matrix[interaction_term, interaction_term]
      cov_base_interaction <- vcov_matrix[base_term, interaction_term]
      
      # Combined variance
      combined_variance <- var_base + var_interaction + 2 * cov_base_interaction
      se <- sqrt(combined_variance)
      
      # Log OR for interaction term
      log_or <- interaction_coef+base_coef
      or <- exp(log_or)
    } else {
      # Without interaction term
      log_or <- base_coef
      or <- exp(log_or)
      se <- sqrt(var_base)
    }
    
    # Confidence intervals
    lower_ci <- round(exp(log_or - 1.96 * se), 2)
    upper_ci <- round(exp(log_or + 1.96 * se), 2)
    result_levels[[level]] = list(or = or, lower_ci = lower_ci, upper_ci = upper_ci)
    }
    # summary_tab=summary(model_clinical_doc)$coefficients
  
  return(bind_rows(result_levels,.id="Level"))
}

# apply sugroup to age_adult, time_of_day, day_type
subgroup_analyses=list()
p_values=list()
for(subgroup in c("age_adult","time_of_day","day_type",'sentinel_condition')){
  formula=paste0("failure_num~arm*",subgroup,"+(1|clinician_num) + (1|hospital_num)")
  model=glmer(formula,data=outcome_data,family=binomial())
  subgroup_analyses[[subgroup]]=calculate_or_ci(model=model,
    base_term="armIntervention",subgroup_name=subgroup)
    interaction_term=grep(paste0("armIntervention:",subgroup),names(fixef(model)),value=T)
    p_values[[subgroup]] <- summary(model)$coefficients[interaction_term, "Pr(>|z|)"]
}

tab_subgroup<-subgroup_analyses %>%
  bind_rows(.id="subgroup") %>%
  mutate(p_value=recode(subgroup,!!!p_values)) %>%
  mutate(subgroup=recode(subgroup,'age_adult'="Age group","time_of_day"="Time of Day","day_type"="Day Type","sentinel_condition"="Sentinel Condition")) %>%
  rename("Odds Ratio of treatment failure"=or,"Lower 95% CI"=lower_ci,"Upper 95% CI"=upper_ci) %>%
  mutate_if(is.numeric,round,2) 
  
tab_subgroup%>%
  gt() %>%
  gt::gtsave(filename = file.path(output_folder, "subgroup_analyses.html")   )


# forrest plot of subgroup analyses
tab_subgroup<-tab_subgroup %>%mutate(subgroup2=glue("{subgroup} (p-value={p_value}*)"))
ggplot(tab_subgroup,aes(y=Level))+
geom_point(aes(x=`Odds Ratio of treatment failure`),size=3)+
facet_wrap(.~subgroup2,scales="free_y",ncol = 1)+
geom_errorbarh(aes(xmin=`Lower 95% CI`,xmax=`Upper 95% CI`),height=0.1)+
geom_text(aes(x=12,label=glue("{`Odds Ratio of treatment failure`} [{`Lower 95% CI`}, {`Upper 95% CI`}]")),hjust=.2,size=3)+
geom_vline(xintercept=1,linetype="dashed")+
labs(x="Odds Ratio of treatment failure (Log scale)",y="")+
scale_x_continuous(trans="log",breaks = c(0.1,0.5,1,2,5,10))+
coord_cartesian(xlim=c(0.1,16.0))+
theme_igray()+
theme(strip.position = "top")

ggsave(filename = file.path(output_folder, "subgroup_analyses_forest_plot.png"),width=10,height=4,dpi=300)

