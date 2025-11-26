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
library(ordinal)





# load environment variables from .env file

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")

outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))
clinical_doc_data<-read_parquet(file.path(data_folder,"CLINICAL_DOCUMENTATION_COLLATED.parquet")) 


df = left_join(clinical_doc_data,outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("study_id"="record_id")) %>% 
mutate(y=min_score,
y_factor=factor(min_score,levels=1:5,labels = c("Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"),ordered=T)) %>% filter(!is.na(y))




desc_ratings=df  %>% group_by(y_factor,arm,variable) %>% summarise(n=n()) %>% 
ungroup() %>% 
group_by(variable,arm)  %>% mutate(N=sum(n)) %>% mutate(perc=n/N*100,
variable=recode(variable,
"appropriate"="Appropriateness of\n Diagnosis",
"comprehensive"="Comprehensiveness of\n Documentation",
"treatment_appropriate"="Appropriateness of\n Proposed Treatment Plan")) 


# plot stacked bar chart of perc by arm and facet by variable
ggplot(desc_ratings,aes(x=arm,y=perc,fill=y_factor)) +
  geom_bar(stat="identity",position=position_stack(reverse=T)) +
  facet_wrap(variable~.,ncol=3,scales="free_y") +
#   scale_y_continuous(labels=scales::percent_format()) +
  scale_fill_brewer(palette="Blues",name="Rating") +
  labs(x="Study Arm",y="Percent") +
  theme_igray() +
  theme(legend.position="bottom")

ggsave(file.path(output_folder,"clinical_documentation_ratings.png"),width=10,height=5)




desc_ratings %>%
pivot_wider(
  names_from = y_factor,
  values_from = perc,
  id_cols = c(variable, arm)
) %>%
  arrange(variable, arm) %>%
  gt() %>%
  gtsave(
    filename = file.path(output_folder, "clinical_documentation_ratings.docx")
  )

desc_ratings %>%
pivot_wider(
  names_from = arm,
  values_from = perc,
  id_cols = c(variable, y_factor)
) %>%
  arrange(variable, y_factor) %>%
  gt() %>%
  gtsave(
    filename = file.path(output_folder, "clinical_documentation_ratings_by_arm.docx")
  )



# ordinal logistic regression with interaction between arm and variable and random intercept for clinician and hospital
library(ordinal)
model_clinical_doc=clmm(y_factor ~ arm*variable + (1|clinician_num) + (1|hospital_num),data=df)
summary(model_clinical_doc)




# compute the odds ratios and 95% CI for arm in each variable

calculate_or_ci <- function(model, base_term, interaction_term = NA) {
    coefs<-coef(model)
    vcov_matrix<-vcov(model)
    # summary_tab=summary(model_clinical_doc)$coefficients
  if (!base_term %in% names(coefs)) {
    stop(paste("Base term not found in coefficients:", base_term))
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
  
  return(list(or = round(or, 2), ci = paste0(lower_ci, ", ", upper_ci)))
}

results=list()

for(var in c("appropriate","comprehensive","treatment_appropriate")){
    result_=calculate_or_ci(model=model_clinical_doc,
    base_term="armIntervention",
    interaction_term=ifelse(var=="appropriate",NA,paste0("armIntervention:variable",var)))
   results[[var]] <- result_
}

bind_rows(results,.id="variable") %>%
mutate(variable=recode(variable,
"appropriate"="Appropriateness of Diagnosis",
"comprehensive"="Comprehensiveness of Documentation",
"treatment_appropriate"="Appropriateness of Proposed Treatment Plan")) %>%
  rename("Odds Ratio"=or,"95% CI"=ci) %>%
  gt() %>%
  gtsave(filename = file.path(output_folder, "clinical_documentation_ordinal_logistic.html")   )


tab_appropriate=df %>% filter(variable=="appropriate") %>%
tbl_summary(include=c("y_factor"),by="arm",label=list(y_factor="Appropriateness of Diagnosis"))  %>% add_overall()

tab_comprehensive=df %>% filter(variable=="comprehensive") %>%
tbl_summary(include=c("y_factor"),by="arm",label=list(y_factor="Comprehensiveness of Documentation")) %>% add_overall()

tab_treatment_appropriate=df %>% filter(variable=="treatment_appropriate") %>%
tbl_summary(include=c("y_factor"),by="arm",label=list(y_factor="Appropriateness of Proposed Treatment Plan")) %>% add_overall()


tab_doc_all=tbl_stack(list(tab_appropriate,tab_comprehensive,tab_treatment_appropriate)) %>%modify_header(label="Domain")
tab_doc_all %>%as_gt() %>%
  gtsave(filename = file.path(output_folder, "clinical_documentation_ratings_summary by arm.html") )