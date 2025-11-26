library(arrow)
library(dplyr)
library(gtsummary)
library(gt)
library(lme4)
library(stringr)
library(janitor)
library(brms)
library(tidybayes)





# load environment variables from .env file

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")

outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))


#   Allow model effects to partially pool across domains via random slopes by domain.
#   This borrows strength across domains while estimating domain-specific model effects.
formulaA <- bf(failure_num ~ 1 + arm_num + (1 + arm_num | hospital_num) + (1| clinician_num))



# Priors (weakly informative) -- adjust to your context
priors <- c(
  prior(normal(0, 1.5), class = "b"),                # coefficients on latent (logit) scale
  prior(normal(0, 10), class = "Intercept"),           # global intercept
  prior(cauchy(0, 2), class = "sd"),          # SDs for group-level effects
  prior(lkj(2), class = "cor")                  # correlation prior for multivariate normal group effects
)

# Fit (example with formulaA). Adjust iter/warmup/chains for your compute.
fit <- brm(
  formula = formulaA,
  data = outcome_data,
  family = bernoulli(link = "logit"),
  prior = priors,
  chains = 4,
  iter = 5000,
  warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  seed = 2025,
  cores = parallel::detectCores()
)

saveRDS(fit, file = file.path(results_folder, "model_random_slope.rds"))

# estimate the effect of arm_num for each hospital

library(ggplot2)
library(forcats)
pp<-fit %>%
  spread_draws(r_hospital_num[ , arm_num]) %>%
  mutate(hospital_num = as.factor(hospital_num)) 

pp %>%   ggplot(aes(x = fct_reorder(hospital_num, r_hospital_num), y = r_hospital_num)) +
  geom_pointinterval() +
  labs(
    x = "Hospital",
    y = "Estimated effect of intervention (log-odds scale)",
    title = "Estimated effect of intervention by hospital",
    subtitle = "Points represent posterior medians; thick and thin lines represent 50% and 95% credible intervals, respectively"
  ) +
  theme_minimal()
ggsave(file.path(output_folder, "random_slope_hospital_effects.png"), width = 8, height = 6)    