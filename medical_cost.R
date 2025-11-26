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
library(brms)





# load environment variables from .env file

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")

outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))

data_cost<-read_excel(file.path(results_folder,"Medical costs - alibayram__medgemma:27b.xlsx"))
# data_cost<-read_parquet(file.path(results_folder,"cost_data.parquet"))


cost_mapping<-list("antibiotics_cost" ="Antibiotics", "antimalarials_cost" ="Antimalarials",     
"antihistamines_cost"="Antihistamines", "analgestics_cost"="Analgesics, Antipyretics and Anti-inflammatories",        
"respiratory_cost" ="Respiratory Medicines (Non-antibiotic, Non-antihistamine)", "gastrointestinal_cost" ="Gastrointestinal Medicines",   
"chronic_cost" ="Chronic Disease Medicines","reproductive_cost"  ="Reproductive Health, STI and Genitourinary Medicines",     
"dermatological_cost"="Dermatological Medicines (skin only)","neurology_cost"="Mental Health and Neurology Medicines",          
"supplement_cost"="Vitamins and Supplements", "supportive_cost"="Miscellaneous and Supportive Treatments")





df <- outcome_data %>%
  left_join(data_cost, by=c("record_id")) %>%
  filter(n_clinicians ==1)  %>% 
  mutate(total_cost=TotalInvoiceAmount)

df <-df %>% 
mutate(total_med_cost = antibiotics_cost + antimalarials_cost + antihistamines_cost + analgestics_cost +
         respiratory_cost + gastrointestinal_cost + chronic_cost + reproductive_cost +
         dermatological_cost + neurology_cost + supplement_cost + supportive_cost )


# average cost by arm
df  %>% 
tbl_summary(include=c("antibiotics_cost","antimalarials_cost","antihistamines_cost","analgestics_cost",
                      "respiratory_cost","gastrointestinal_cost","chronic_cost","reproductive_cost",
                      "dermatological_cost","neurology_cost","supplement_cost","supportive_cost"),
             statistic = list(all_continuous() ~ "{mean} ({sd})"),
             digits = all_continuous() ~ 1,
             by = arm,
             missing = "no",
             label=cost_mapping) %>%
  modify_header(label = "**Medication category**") %>%add_overall() %>%
  as_gt() %>%
  gt::gtsave(
    filename = file.path(output_folder, "medical_cost_breakdown_summary.html")
  )

df  %>% tbl_summary(include=c("total_med_cost","investigations_fees","TotalInvoiceAmount"),
             statistic = list(all_continuous() ~ "{mean} ({sd})"),
             digits = all_continuous() ~ 1,
             by = arm,
             missing = "no",
             label = list(total_med_cost="Total Medication Cost",
                          investigations_fees="Investigations Fees",
                          TotalInvoiceAmount="Total fees"
                          )) %>%
  modify_header(label = "**Variable**") %>%add_overall() %>%
  as_gt() %>%
  gt::gtsave(
    filename = file.path(output_folder, "medical_cost_overview_summary.html")
  )


ggplot(df, aes(x=total_cost,fill=arm)) +
  geom_density( color="black", alpha=0.3) +
  labs(title="Distribution of Total Medical Costs",
       x="Total Medical Cost",
       y="Density") +
  theme_minimal()





df  %>% 
tbl_summary(include = c(total_cost, arm),
             statistic = list(total_cost ~ "{mean} ({sd})"),
             by = arm,
             missing = "no") %>%
  modify_header(label = "**Variable**") %>%
  as_gt() %>%
  gt::gtsave(
    filename = file.path(output_folder, "medical_cost_summary.html")
  )


model<-lmer(total_cost ~ arm + (1|hosp_id) + (1|clinician_num),
data = df)

summary(model)

tab_regression_total <- tbl_regression(
  model,add_estimate_to_reference_rows=F,
  exponentiate = FALSE,
  label = list(arm = "Study Arm")
)# %>%
  # as_gt() %>%
  # gt::gtsave(
  #   filename = file.path(output_folder, "medical_cost_model_results.html")
  # )


fit_medications_total<-lmer(total_med_cost~arm + (1| hosp_id) + (1|clinician_num),data=df)

tab_regression_medicines <- tbl_regression(
  fit_medications_total,add_estimate_to_reference_rows=F,
  exponentiate = FALSE,
  label = list(arm = "Study Arm")
) 

fit_investigations_total<-lmer(investigations_fees~arm + (1| hosp_id) + (1|clinician_num),data=df)
tab_regression_investigations<-tbl_regression(fit_investigations_total,add_estimate_to_reference_rows=F,
  exponentiate = FALSE,
  label = list(arm = "Study Arm"))

tbl_merge(list(tab_regression_total,tab_regression_medicines,tab_regression_investigations),
tab_spanner = c("**Total cost**","**Medications cost**","**Investigations cost**")) %>% 
as_gt() %>% 
gtsave(file.path(output_folder,"Models for total, medication, and investigations.html"))

df2<-df %>% select(record_id,arm,hosp_id,clinician_num,antibiotics_cost,antimalarials_cost,antihistamines_cost,analgestics_cost,
                  respiratory_cost,gastrointestinal_cost,chronic_cost,reproductive_cost,
                  dermatological_cost,neurology_cost,supplement_cost,supportive_cost) %>%
pivot_longer(cols=c("antibiotics_cost","antimalarials_cost","antihistamines_cost","analgestics_cost",
                      "respiratory_cost","gastrointestinal_cost","chronic_cost","reproductive_cost",
                      "dermatological_cost","neurology_cost","supplement_cost","supportive_cost"),
             names_to="category",
             values_to="cost") %>%
mutate(category = recode(category, !!!cost_mapping))


fit0<-lmer(cost ~ arm * category + (1|hosp_id) + (1|clinician_num),
data = df2)




calculate_mean_ci <- function(model, base_term,subgroup_name) {
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
      mean_diff <- interaction_coef+base_coef
    } else {
      # Without interaction term
      mean_diff <- base_coef
      se <- sqrt(var_base)
    }
    
    # Confidence intervals
    lower_ci <- round(mean_diff - 1.96 * se, 2)
    upper_ci <- round(mean_diff + 1.96 * se, 2)
    result_levels[[level]] = list(mean_diff = mean_diff, lower_ci = lower_ci, upper_ci = upper_ci)
    }
    # summary_tab=summary(model_clinical_doc)$coefficients
  
  return(bind_rows(result_levels,.id="Level"))
}

subgroup_results <- calculate_mean_ci(fit0,"armIntervention","category") %>% 
mutate(Level=recode(Level, !!!cost_mapping)) %>% 
rename(`Medication Category`=Level,
       `Mean Difference in Cost (KSH)`=mean_diff,
       `95% CI Lower`=lower_ci,
       `95% CI Upper`=upper_ci)

subgroup_results %>%
  gt() %>% fmt_number(
    decimals = 1
  ) %>%
  gt::gtsave(
    filename = file.path(output_folder, "medical_cost_subgroup_results.html")
  )

fit <- brm(
  bf(cost ~ arm * category + (1 | hosp_id) + (1 | clinician_num),
     hu ~ arm + category),
  data = df2,
  family = hurdle_gamma(),
  chains = 4, cores = 4
)