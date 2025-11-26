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
clinician_data<-read.csv(file.path(data_folder,'Clinician data.csv'),check.names=FALSE) %>%
  filter(`Clinician ID` %in% outcome_data$clinician_id)

operational_data=read_parquet(file.path(data_folder, 'Operational_data.parquet'))

diagnoses_data=read_excel(file.path(results_folder,"Body system - alibayram__medgemma:27b.xlsx"))

#merge outcome_data with diagnoses data
outcome_data<-left_join(outcome_data,diagnoses_data %>% select(record_id,cardiovascular:unspecified_other),by='record_id')
outcome_data<-outcome_data %>% 
mutate(across(cardiovascular:unspecified_other,as.numeric))

#merge outcome_data with clinician data to get years of experience
clinician_data<-clinician_data %>% 
mutate(clinician_id=sprintf("%.0f",`Clinician ID`)) %>% 
mutate(years_penda=factor(years_penda,levels=c("<3 months","3mths-1yrs","1-2 yrs","2-5 yrs", "5+ yrs"),ordered=TRUE),
years_penda=factor(years_penda,ordered=F)) 
outcome_data<-outcome_data %>%
left_join(clinician_data %>% select(`clinician_id`,`years_penda`,`years_exp`),
          by=c("clinician_id"="clinician_id")) 

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


#################################################################################################################################################3
#
# DESCRIPTIVE ANALYSES
#

###################################################################################
# patient characteristics
outcome_data %>% tbl_summary(by=arm,include = c("age_category","gender","hosp_id"),percent="column",
                             label=list("age_category"="Age Category","gender"="Gender","hosp_id"="Hospital")) %>%
  add_overall() %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "baseline_characteristics.html"))

# Clinician characteristics

clinician_characteristics=clinician_data %>% 
tbl_summary(by="arm",include=c("years_exp","years_penda"),label=list("years_exp"="Years of experience","years_penda"="Duration at Penda ")) %>% add_overall() 
as_gt(clinician_characteristics) %>%
  gt::gtsave(filename = file.path(output_folder, "clinician_characteristics.html")   )

# diagnoses
diag_map=list("cardiovascular"="Cardiovascular","dermatologic"="Dermatologic","ent_dental_ophthalmologic"="ENT, Dental, Ophthalmologic" ,
"febrile_infectious"= "Febrile / Infectious","gastrointestinal"="Gastrointestinal","genitourinary_reproductive"="Genitourinary & Reproductive",
 "musculoskeletal"="Musculoskeletal" ,"neurologic_psychiatric"="Neurologic & Psychiatric","respiratory"="Respiratory","unspecified_other"="Unspecified / Other")

diags<-outcome_data %>% select(record_id,hosp_id,arm,cardiovascular:unspecified_other) %>% 
pivot_longer(cols=cardiovascular:unspecified_other) %>% 
mutate(name=recode(name, !!!diag_map))

diags2<-diags %>% group_by(hosp_id,name) %>% 
summarise(prop=mean(value)*100) %>% 
mutate(prop_display=if_else(round(prop,2)<0.1,"<0.1",sprintf("%.1f",prop))) %>%
ungroup()


ggplot(diags2)+
geom_tile(aes(y=hosp_id,x=name,fill=prop))+
geom_text(aes(y=hosp_id,x=name,label=prop_display), size=3)+
scale_fill_gradient("Prevalence (%)",low="grey90",high = "grey10")+
theme_minimal()+
theme(axis.text.x = element_text(angle = -25, vjust = 0, hjust = 0))+
labs(x="",y="Center")

ggsave(filename = file.path(output_folder,"Prevalence of diagnoses by hospital.png"))


outcome_data %>% 
tbl_summary(by=arm,include=c("cardiovascular","dermatologic","ent_dental_ophthalmologic",
"febrile_infectious","gastrointestinal","genitourinary_reproductive",
"musculoskeletal","neurologic_psychiatric","respiratory","unspecified_other"),
label=diag_map,percent="column") %>% add_overall() %>%
as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "diagnoses_by_study_arm.html")   )


#################################################################################################################################################3
#
# MULTILEVEL LOGISTIC REGRESSION ANALYSES
#

###################################################################################


# multilevel models

mod0=glmer(failure_num~arm+(1|clinician_id)+(1|hosp_id),
                       data=outcome_data,# |> filter(!study_id %in% c("550290","420205")),
                       family=binomial())

mod0_gt=tbl_regression(mod0,
                              exponentiate = T,label=list(arm="Study Arm"))
tab0=tbl_summary(outcome_data,
                   include=c("arm"),by="failure",percent="row",
                   digits=list(arm=c(0,1)),label=list(arm="Study Arm"))

tbl_merge(
  tbls = list(tab0,mod0_gt),
  tab_spanner = c("","**Multilevel logistic regression**")
) %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "model_primary_multilevel.html")     )



mod0_pp=glmer(failure_num~arm+(1|clinician_id)+(1|hosp_id),
                       data=outcome_data |> filter(violation==F),
                       family=binomial())

mod0_gt_pp=tbl_regression(mod0_pp,
                              exponentiate = T,label=list(arm="Study Arm"))
tab0_pp=tbl_summary(outcome_data|> filter(violation==F),
                   include=c("arm"),by="failure",percent="row",
                   digits=list(arm=c(0,1)),label=list(arm="Study Arm"))

tbl_merge(
  tbls = list(tab0_pp,mod0_gt_pp),
  tab_spanner = c("","**Multilevel logistic regression**")
) %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "model_primary_multilevel_per_protocol.html")     )


# model0=glmer(failure_num~arm_num+(1|clinician_num) + (1|hospital_num),
#                      data=outcome_data,
#                      family=binomial())

# tbl_regression(model0, exponentiate=TRUE,label=list("arm_num"="Intervention")) %>%
#   as_gt() %>%
#   gt::gtsave(filename = file.path(output_folder, "model_primary_multilevel.html")   )




# tbl_regression(model0, exponentiate=TRUE,label=list("arm_num"="Intervention")) %>%
#   as_gt() %>%
#   gt::gtsave(filename = file.path(output_folder, "model_primary_multilevel.html")   )



model1=glmer(failure_num~arm + age_category2 + years_exp + years_penda + gender+
cardiovascular+dermatologic+ent_dental_ophthalmologic+febrile_infectious+gastrointestinal+
genitourinary_reproductive+musculoskeletal+neurologic_psychiatric+respiratory+
unspecified_other+day_type+(1|hosp_id)+(1|clinician_num),
                     data=outcome_data,
                     family=binomial(),
                     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))




lab_list=list("arm"="Study Arm","age_category2"="Age Category","years_exp"="Years of Experience",
"years_penda"="Years at Penda","gender"="Gender","cardiovascular"="Cardiovascular","dermatologic"="Dermatologic",
"ent_dental_ophthalmologic"="ENT, Dental, Ophthalmologic" ,
"febrile_infectious"= "Febrile / Infectious","gastrointestinal"="Gastrointestinal","genitourinary_reproductive"="Genitourinary & Reproductive",
 "musculoskeletal"="Musculoskeletal" ,"neurologic_psychiatric"="Neurologic & Psychiatric","respiratory"="Respiratory",
 "unspecified_other"="Unspecified / Other","day_type"="Day Type","hosp_id"="Centre")
tbl_regression(model1, exponentiate=TRUE,label=lab_list) %>%  
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "model_adjusted_multilevel.html")   )



# model_interactions=glmer(failure_num~arm_num*as.factor(hospital_num)+(1|clinician_num) ,
#                      data=outcome_data,
#                      family=binomial())

# tbl_regression(model_interactions,exponentiate=TRUE) %>%
#   as_gt() %>%
#   gt::gtsave(filename = file.path(output_folder, "model_interactions.html")   )

#################################################################################################################################################3
#
# CO-PRIMARY OUTCOMES ANALYSES
#

###################################################################################


model_coprimary=glmer(outcome_secondary~arm+(1|clinician_num) + (1|hospital_num),
                     data=outcome_data,
                     family=binomial())

tbl_coprimary=tbl_regression(model_coprimary, exponentiate=TRUE,label=list("arm"="Study Arm")) 

tab_coprimary=tbl_summary(outcome_data,
                   include=c("arm"),by="outcome_secondary",percent="row",
                   digits=list(arm=c(0,1)),label=list(arm="Study Arm"))

tbl_merge(
  tbls = list(tab_coprimary,tbl_coprimary),
  tab_spanner = c("**Death/Hospitalization**","**Multilevel logistic regression**")
) %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "model_coprimary_multilevel.html")     )

#################################################################################################################################################3
#
# ADJUSTED BAYESIAN ANALYSES
#

###################################################################################
## multilevel logistic regression model using brms

prior_brms <- c(
    prior(normal(0, 0.8), class = "b"),         # treatment effect approx OR ~ exp(N(0,0.8))
    prior(student_t(3, 0, 0.5), class = "sd"),  # tight prior on clinician SD
    prior(student_t(3, 0, 5), class = "Intercept")
  )
weak_priors= c(
  prior(normal(0, 10), class = "b"),                # coefficients on latent (logit) scale
  prior(normal(0, 10), class = "Intercept"),           # global intercept
  prior(exponential(0.5), class = "sd")          # SDs for group-level effects
) 
model_brms=brm(failure_num~arm + age_category2 + years_exp + years_penda + gender+
cardiovascular+dermatologic+ent_dental_ophthalmologic+febrile_infectious+gastrointestinal+
genitourinary_reproductive+musculoskeletal+neurologic_psychiatric+respiratory+
unspecified_other+day_type+
(1|hosp_id)+
(1|clinician_num),
                       data=outcome_data,
                       family=bernoulli(link="logit"),
                       prior=weak_priors,
                       control = list(adapt_delta = 0.995, max_treedepth = 15),
                       chains=4,iter=4000,warmup=1000,cores=4,
                       seed=1234,
                       silent=TRUE,
                       refresh=0
                       )

#save model to disk
saveRDS(model_brms,file=file.path(results_folder,"logistic_primary_adjusted_brms.rds"))
# model_brms=readRDS(file.path(results_folder,"logistic_primary_adjusted_brms.rds"))

# format outputs of model_brms for publication. include odds ratios and 95% credible intervals
results_brms <- model_brms %>%
  gather_draws(`b_.*`,regex = T) %>%
  median_qi(.value = exp(.value)) %>%
  mutate(
    Variable = str_remove(.variable, "^b_"),
    Estimate = round(.value, 2),
    Lower_95_CrI = round(.lower, 2),
    Upper_95_CrI = round(.upper, 2)
  ) %>%
  select(Variable, Estimate, Lower_95_CrI, Upper_95_CrI)

# estimate risk difference from brms model conditional on treatment group over the covariate distribution.
new_dat_intervention<-outcome_data %>%
  mutate(arm="Intervention")
new_dat_control<-outcome_data %>%
  mutate(arm="Control")
pred_intervention<-posterior_epred(model_brms,newdata=new_dat_intervention,re_formula=NA)
pred_control<-posterior_epred(model_brms,newdata=new_dat_control,re_formula=NA)

# Average probability of failure in arm=1 over the covariate distribution
avg_prob_intervention <- rowMeans(pred_intervention)

# Average probability of failure in arm=0 over the covariate distribution
avg_prob_control <- rowMeans(pred_control)

# The Risk Difference for each MCMC draw
posterior_risk_difference <- avg_prob_intervention - avg_prob_control

# Calculate the Point Estimate (Posterior Mean)
amrd_estimate <- median(posterior_risk_difference)

# Calculate the 95% Equal-Tailed Interval (ETI)
amrd_ci_eti <- quantile(posterior_risk_difference, probs = c(0.025, 0.975))

#predicted risk by study arm and 95% credible interval
pred_summary <- data.frame(
  Group = c("Intervention", "Control"),
  `Predicted Risk` = c(median(avg_prob_intervention), median(avg_prob_control)),
  `Lower 95% CrI` = c(quantile(avg_prob_intervention, 0.025), quantile(avg_prob_control, 0.025)),
  `Upper 95% CrI` = c(quantile(avg_prob_intervention, 0.975), quantile(avg_prob_control, 0.975))
)
pred_summary<-bind_rows(pred_summary,
                        data.frame(
                          Group="Risk difference (Intervention - Control)",
                          `Predicted Risk`=amrd_estimate,
                          `Lower 95% CrI`=amrd_ci_eti[1],
                          `Upper 95% CrI`=amrd_ci_eti[2]
                        )
) 

pred_summary %>%
  gt() %>%
  fmt_number(columns=2:4, decimals=3) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(rows = 3)
  ) %>%
  gt::gtsave(filename = file.path(output_folder, "predicted_risk_brms.html"))


ggplot(data.frame(risk_diff = posterior_risk_difference), aes(x = risk_diff)) +
  geom_density(fill = "skyblue", alpha = 0.4) +
  geom_vline(aes(xintercept = mean(risk_diff)), color = "blue", linetype = "dashed") +
  geom_vline(aes(xintercept = quantile(risk_diff, c(0.025))),
             color = "red", linetype = "dotted")+
  geom_vline(aes(xintercept = quantile(risk_diff, c(0.975))),
             color = "red", linetype = "dotted")+

  labs(x = "Population-level risk difference", y = "Posterior density")


#################################################################################################################################################3
#
# META ANALYSES
#

###################################################################################



# simpler brms model (random slope for treatment arm)
prior_random_slope <-  c(
  prior(normal(0, 1), class = "b"),                # coefficients on latent (logit) scale
  prior(normal(0, 10), class = "Intercept"),           # global intercept
  prior(cauchy(0, 1), class = "sd"),          # SDs for group-level effects
  prior(lkj(2), class = "cor")                  # correlation prior for multivariate normal group effects
)
mod_random_slope<-brm(
  formula = failure_num ~ arm + (1 + arm | hosp_id) + (1 | clinician_num),
  data = outcome_data,
  family = bernoulli(link = "logit"),
  prior = prior_random_slope,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  chains = 4, iter = 4000, warmup = 1000, cores = 4,
  seed = 1234,
  silent = TRUE,
  refresh = 0
)


# save model to disk
saveRDS(mod_random_slope,file=file.path(results_folder,"logistic_random_slope_brms.rds"))
# mod_random_slope=readRDS(file.path(results_folder,"logistic_random_slope_brms.rds"))

study.draws <- spread_draws(mod_random_slope, r_hosp_id[hosp,], b_armIntervention) %>% 
  mutate(b_armIntervention = r_hosp_id + b_armIntervention)

pooled.effect.draws <- spread_draws(mod_random_slope, b_armIntervention) %>% 
  mutate(hosp = "Pooled Effect")

forest.data <- bind_rows(study.draws, 
                         pooled.effect.draws) %>% 
   ungroup() %>%
   mutate(hosp = str_replace_all(hosp, "[.]", " ")) %>% 
   mutate(hosp = reorder(hosp, b_armIntervention)) %>% 
   mutate(b_armIntervention = exp(b_armIntervention))

forest.data.summary <- group_by(forest.data, hosp) %>% 
  median_qi(b_armIntervention) 

ggplot(aes(b_armIntervention, 
           relevel(hosp, "Pooled Effect", 
                   after = Inf)), 
       data = forest.data) +
  
  # Add vertical lines for pooled effect and CI
  geom_vline(xintercept = exp(fixef(mod_random_slope)[2, 1]), 
             color = "grey", size = 1) +
  geom_vline(xintercept = exp(fixef(mod_random_slope)[2, 3:4]), 
             color = "grey", linetype = 2) +
  geom_vline(xintercept = 1, color = "black",size = 1) +
  #            stat_halfeye(.width = .95, size = 3/3,alpha=0.8)+
  
  # # # Add densities
  geom_density_ridges(fill = "grey", 
                      rel_min_height = 0.01, 
                      col = NA, scale = 1,
                      alpha = 0.8) +
  geom_pointinterval(data = forest.data.summary, aes(xmin = .lower, xmax = .upper),
                      size = 1) +
  
  # Add text and labels
  geom_text(data = mutate_if(forest.data.summary, 
                             is.numeric, round, 2),
    aes(label = glue("{b_armIntervention} [{.lower}, {.upper}]"), 
        x = Inf), hjust = "inward",vjust = -1,size=2,fontface="bold") +
        coord_cartesian(xlim = c(0.1, 3.0)) +
  labs(x = "Odds Ratio", # summary measure
       y = element_blank()) +
  theme_minimal()+
   theme(panel.grid.x   = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y  = element_text(hjust = 0))
ggsave(file.path(output_folder, "meta_analysis_forest_plot.png"),width=8,height=5,dpi=300)



# # fit separate brms models by hospital

# priors_hosp <- c(
#     prior(student_t(7,0, 0.35), class = "b"),         # treatment effect approx OR ~ exp(-0.7,0.7)
#     prior(student_t(3, 0, 0.2), class = "sd"),  # tight prior on clinician SD
#     prior(student_t(4, 0, 10), class = "Intercept")
#   )

# hosp_models_brms <- list()

# for (hosp in unique(outcome_data$hosp_id)) {
  
#   data_hosp <- outcome_data %>% filter(hosp_id == hosp)
  
#   # Fit Bayesian logistic mixed model
#   model_hosp <- brm(
#     formula = failure_num ~ arm + (1 | clinician_num),
#     prior = priors_hosp,
#     data = data_hosp,
#     family = bernoulli(link = "logit"),
#     control = list(adapt_delta = 0.995, max_treedepth = 15),
#     chains = 4, iter = 4000, warmup = 1000, cores = 4,
#     seed = 1234,
#     silent = TRUE,
#     refresh = 0
#   )
#   hosp_models_brms[[as.character(hosp)]] <- model_hosp
# }

# # save models to disk
# saveRDS(hosp_models_brms,file=file.path(results_folder,"logistic_hosp_models_brms.rds"))
# # hosp_models_brms <- readRDS(file.path(results_folder,"logistic_hosp_models_brms.rds"))

# hosp_estimates_brms <- list()
# for (hosp in unique(outcome_data$hosp_id)) {
#   model_hosp <- hosp_models_brms[[as.character(hosp)]]
  
#   # Extract posterior summaries for fixed effects and SDs
#   post <- as_draws_df(model_hosp)
  
#   estimates <- list(
#     intercept_mean = mean(post$b_Intercept),
#     intercept_sd   = sd(post$b_Intercept),
#     est       = mean(post$b_armIntervention),
#     sd         = sd(post$b_armIntervention),
#     arm_OR_mean    = mean(exp(post$b_armIntervention)),
#     arm_OR_CI_low  = quantile(exp(post$b_armIntervention), 0.025),
#     arm_OR_CI_high = quantile(exp(post$b_armIntervention), 0.975),
#     sd_clinician_mean = mean(post$sd_clinician_num__Intercept),
#     sd_clinician_sd   = sd(post$sd_clinician_num__Intercept)
#   )
  
#   # Add counts
#   estimates[["nclinician_intervention"]] <-
#     outcome_data %>% filter(arm == "Intervention" & hosp_id == hosp) %>% pull(clinician_id) %>% unique() %>% length()
#   estimates[["nclinician_control"]] <-
#     outcome_data %>% filter(arm == "Control" & hosp_id == hosp) %>% pull(clinician_id) %>% unique() %>% length()
#   estimates[["n_obs"]] <- nrow(outcome_data %>% filter(hosp_id == hosp))
#   estimates[["events_intervention"]]<-outcome_data %>% filter(arm == "Intervention" & hosp_id == hosp) %>% pull(failure_num) %>% sum()
#   estimates[["events_control"]]<-outcome_data %>% filter(arm == "Control" & hosp_id == hosp) %>% pull(failure_num) %>% sum()
# estimates[["n_intervention"]]<-outcome_data %>% filter(arm == "Intervention" & hosp_id == hosp) %>% nrow()
# estimates[["n_control"]]<-outcome_data %>% filter(arm == "Control" & hosp_id == hosp) %>% nrow()
#   hosp_estimates_brms[[as.character(hosp)]] <- estimates
# }



# hosp_brms_data <- bind_rows(hosp_estimates_brms, .id = "hosp_id")



# # hosp_estimates=list()
# # for(hosp in unique(outcome_data$hosp_id)){
# #   model_hosp=brm(failure_num~arm+(1|clinician_num),
# #                        data=outcome_data %>% filter(hosp_id==hosp),
# #                        family=binomial())
# #   estimates=get_est_sd_glmer(model_hosp)
# #   estimates[['nclinician_intervention']]=outcome_data %>% filter(hosp_id==hosp & arm=="Intervention") %>% pull(clinician_id) %>% unique() %>% length()
# #   estimates[['nclinician_control']]=outcome_data %>% filter(hosp_id==hosp & arm=="Control") %>% pull(clinician_id) %>% unique() %>% length()
# #   estimates[['n_obs']]=outcome_data %>% filter(hosp_id==hosp) %>% nrow()
# #   hosp_estimates[[as.character(hosp)]] <- estimates
# # }

# # hosp_estimates_data<-bind_rows(hosp_estimates,.id="hosp_id") 

# # meta analysis using brms
# priors_meta <- c(prior(normal(0,100), class = "Intercept"),
#             prior(cauchy(0,1.0), class = "sd"))
# meta_model_brms=brm(
#   formula = est | se(sd) ~ 1 + (1 | hosp_id),
#   data =  hosp_brms_data,
#   family = gaussian(),
#   prior = priors_meta,
#   control = list(adapt_delta = 0.99, max_treedepth = 15),
#   chains = 4, iter = 4000, warmup = 1000, cores = 4,
#   seed = 1234,
#   silent = TRUE,
#   refresh = 0
# )


# summary(meta_model_brms)
# pp_check(meta_model_brms)
# ranef(meta_model_brms)

# study.draws <- spread_draws(meta_model_brms, r_hosp_id[hosp,], b_Intercept) %>% 
#   mutate(b_Intercept = r_hosp_id + b_Intercept)

# pooled.effect.draws <- spread_draws(meta_model_brms, b_Intercept) %>% 
#   mutate(hosp = "Pooled Effect")

# forest.data <- bind_rows(study.draws, 
#                          pooled.effect.draws) %>% 
#    ungroup() %>%
#    mutate(hosp = str_replace_all(hosp, "[.]", " ")) %>% 
#    mutate(hosp = reorder(hosp, b_Intercept)) %>% 
#    mutate(b_Intercept = exp(b_Intercept))

# forest.data.summary <- group_by(forest.data, hosp) %>% 
#   median_qi(b_Intercept) 

# ggplot(aes(b_Intercept, 
#            relevel(hosp, "Pooled Effect", 
#                    after = Inf)), 
#        data = forest.data) +
  
#   # Add vertical lines for pooled effect and CI
#   geom_vline(xintercept = exp(fixef(meta_model_brms)[1, 1]), 
#              color = "grey", size = 1) +
#   geom_vline(xintercept = exp(fixef(meta_model_brms)[1, 3:4]), 
#              color = "grey", linetype = 2) +
#   geom_vline(xintercept = 1, color = "black",size = 1) +
#              stat_halfeye(.width = .95, size = 3/3,alpha=0.8)+
  
#   # # # Add densities
#   # geom_density_ridges(fill = "grey", 
#   #                     rel_min_height = 0.01, 
#   #                     col = NA, scale = 1,
#   #                     alpha = 0.8) +
#   # geom_pointinterval(data = forest.data.summary, aes(xmin = .lower, xmax = .upper),
#   #                     size = 1) +
  
#   # Add text and labels
#   geom_text(data = mutate_if(forest.data.summary, 
#                              is.numeric, round, 2),
#     aes(label = glue("{b_Intercept} [{.lower}, {.upper}]"), 
#         x = Inf), hjust = "inward") +
#         coord_cartesian(xlim = c(0.4, 2.0)) +
#   labs(x = "Odds Ratio", # summary measure
#        y = element_blank()) +
#   theme_minimal()+
#    theme(panel.grid   = element_blank(),
#         axis.ticks.y = element_blank(),
#         axis.text.y  = element_text(hjust = 0))
# ggsave(file.path(output_folder, "meta_analysis_forest_plot.png"),width=8,height=5)


