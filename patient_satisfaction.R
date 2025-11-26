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
satisfaction_data<-read_parquet(file.path(data_folder,"user_satisfaction.pq")) 
operational_data<-read_parquet(file.path(data_folder,'Operational_data.parquet'))



consult_length_all=operational_data %>% select(record_id,provider_dulation) %>%
 right_join(outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num,age_category,gender),by=c("record_id"="record_id"))  %>% 
 filter(n_clinicians==1)


tbl_summary(consult_length_all,include=c("provider_dulation"),
    by = arm,
    statistic = list(
      all_continuous() ~ "{mean} ({sd})",
      provider_dulation ~ "{median} ({p25}, {p75})"
    ),
    digits = all_continuous() ~ 1,
    label = list(
      provider_dulation="Duration of consultation in minutes"
    )
  )  %>% add_difference(provider_dulation ~ "wilcox.test") %>%
  add_overall() %>%
  # add_p() %>%
  bold_labels() %>%
  as_gt() %>%
  gtsave(filename = file.path(output_folder, "consultation_length_summary.html"))


model_consult_length=lmer(provider_dulation ~ arm + (1|clinician_num) + (1|hospital_num),data=consult_length_all)
summary(model_consult_length)
tbl_regression(model_consult_length,exponentiate = F) %>%
  as_gt() %>%
  gtsave(filename = file.path(output_folder, "consultation_length_mixed_effects.html")   )



consult_length_data=satisfaction_data %>% select(record_id,consult_length,steps_explained,consult_comprehensive  ,      
concerns_addressed,recommend_others) %>%
 left_join(outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num,age_category,gender),by=c("record_id"="record_id"))  %>% 
 left_join(operational_data %>% select(record_id,provider_dulation),by='record_id') %>% 
 filter(n_clinicians==1,!is.na(consult_length),consult_length!="") %>% 
 mutate(consult_length=recode_factor(consult_length,`1`="Too long",`2`="Just right",`3`="Too short",.ordered = T))  %>% 
 mutate(across(c(steps_explained,consult_comprehensive,concerns_addressed,recommend_others),~as.numeric(.) ))




consult_length_data %>%
  tbl_summary(include=c("age_category","gender",'provider_dulation',"consult_length","steps_explained","consult_comprehensive",      
"concerns_addressed","recommend_others"),
    by = arm,
    type=c(steps_explained, consult_comprehensive, concerns_addressed, recommend_others) ~ "continuous",
    statistic = list(
      all_continuous() ~ "{mean} ({sd})",
      c(steps_explained, consult_comprehensive, concerns_addressed, recommend_others,provider_dulation) ~ "{median} ({p25}, {p75})"
    ),
    digits = all_continuous() ~ 1,
    label = list(age_category="Age Category",
    gender="Gender",
      consult_length = "Consultation Length",
      steps_explained = "Next steps in my management were clearly explained to me",
      consult_comprehensive = "My consultation was thorough and comprehensive",
      concerns_addressed = "My concerns were adequately addressed",
      recommend_others = "I would recommend this clinic to others",
      provider_dulation="Duration of consultation in minutes"
    )
  ) %>%
  add_overall() %>%
  # add_p() %>%
  bold_labels() %>%
  as_gt() %>%
  gtsave(filename = file.path(output_folder, "patient_satisfaction_summary.html"))



satisfaction_long=satisfaction_data %>% select("record_id","steps_explained","consult_comprehensive",      
"concerns_addressed","recommend_others") %>%
pivot_longer(
    cols= c("steps_explained","consult_comprehensive",      
"concerns_addressed","recommend_others"),names_to = "variable",values_to = "value"
)




df = left_join(satisfaction_long,outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("record_id"="record_id")) %>% 
mutate(y=value,
y_factor=factor(value,levels=1:5,labels = c("Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"),ordered=T)) %>% filter(!is.na(y), n_clinicians==1, !is.na(y_factor))


desc_ratings=df  %>% group_by(y_factor,arm,variable) %>% summarise(n=n()) %>% 
ungroup() %>% 
group_by(variable,arm)  %>% mutate(N=sum(n)) %>% mutate(perc=n/N*100,
variable=recode(variable,
"steps_explained"="Next steps in my \nmanagement were clearly\n explained to me",
"consult_comprehensive"="My consultation \nwas thorough and\n comprehensive"  ,      
"concerns_addressed"="My concerns were\n adequately addressed",
"recommend_others"="I would recommend \nthis clinic to\n others")) 


# plot stacked bar chart of perc by arm and facet by variable
ggplot(desc_ratings,aes(x=arm,y=perc,fill=y_factor)) +
  geom_bar(stat="identity",position=position_stack(reverse=T)) +
  facet_wrap(variable~.,ncol=4,scales="free_y") +
#   scale_y_continuous(labels=scales::percent_format()) +
  scale_fill_brewer(palette="Blues",name="Rating") +
  labs(x="Study Arm",y="Percent") +
  theme_igray() +
  theme(legend.position="bottom")

ggsave(file.path(output_folder,"patient_satisfaction_ratings.png"),width=10,height=5)


# desc_ratings %>%
# pivot_wider(
#   names_from = y_factor,
#   values_from = perc,
#   id_cols = c(variable, arm)
# ) %>%
#   arrange(variable, arm) %>%
#   gt() %>%
#   gtsave(
#     filename = file.path(output_folder, "patient_satisfaction_ratings.docx")
#   )


# ordinal logistic regression with interaction between arm and variable and random intercept for clinician and hospital

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

for(var in c("steps_explained","consult_comprehensive"  , "concerns_addressed","recommend_others")){
    result_=calculate_or_ci(model=model_clinical_doc,
    base_term="armIntervention",
    interaction_term=ifelse(var=="concerns_addressed",NA,paste0("armIntervention:variable",var)))
   results[[var]] <- result_
}

bind_rows(results,.id="variable") %>%
mutate(variable=recode(variable,
"steps_explained"="Next steps in my management were clearly explained to me",
"consult_comprehensive"="My consultation was thorough and comprehensive"  ,      
"concerns_addressed"="My concerns were adequately addressed",
"recommend_others"="I would recommend this clinic to others")) %>%
  rename("Odds Ratio"=or,"95% CI"=ci) %>%
  gt() %>%
  gtsave(filename = file.path(output_folder, "patient_satisfaction_ordinal_logistic.html")   )


df2=df  %>%  group_by(record_id,arm,hosp_id,clinician_num) %>% 
summarise(score=mean(as.numeric(y)),
score_factor=factor(round(score),levels=1:5,labels = c("Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"),ordered=T),.groups="drop")

model_clinical_doc2=clmm(score_factor ~ arm + (1|clinician_num) + (1|hosp_id),data=df2)
summary(model_clinical_doc2)

tbl_regression(model_clinical_doc2,exponentiate = T) %>%
  as_gt() %>%
  gtsave(filename = file.path(output_folder, "patient_satisfaction_overall_ordinal_logistic.html")   )