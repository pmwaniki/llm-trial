library(tidyverse)
library(ggplot2)
library(arrow)
library(gtsummary)
library(gt)
library(readxl)

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")

outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))

consult_data=read_parquet(file.path(data_folder,"ai_consult.parquet"))

consult_tokens=consult_data %>% 
group_by(VisitCode) %>% 
summarise(PromptTokens=sum(PromptTokens,na.rm = T),CompletionTokens=sum(CompletionTokens,na.rm = T))

consult_tokens <- consult_tokens %>% 
mutate(input_cost = PromptTokens * 2.5 / 1e6,
       output_cost = CompletionTokens * 10 / 1e6,
       total_cost = input_cost + output_cost)

# remove non digit characters from VisitCode
consult_tokens <- consult_tokens %>% 
mutate(VisitCode = str_remove_all(VisitCode, "[^0-9]"))



outcome_data<-outcome_data %>% 
filter(n_clinicians==1 & arm=="Intervention") %>% 
mutate(visit_number = str_remove_all(visit_number, "[^0-9]"))

df <- outcome_data %>% 
left_join(consult_tokens, by=c("visit_number"="VisitCode"))


ggplot(df, aes(x=total_cost)) +
  geom_density( color="black", fill="blue", alpha=0.3) +
  labs(title="Distribution of LLM Consultation Costs",
       x="LLM Consultation Cost (USD)",
       y="Density") +
  theme_minimal()


df %>% tbl_summary(include = c(total_cost),
             statistic = list(total_cost ~ "{mean} ({sd})"),
             missing = "no",label = list(total_cost = "Total Cost (USD)")) %>%add_ci() %>% 
  modify_header(label = "**Variable**") %>%
  as_gt() %>%
  gt::gtsave(
    filename = file.path(output_folder, "llm_cost_summary.html")
    )