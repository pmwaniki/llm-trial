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
library(networkD3)




# load environment variables from .env file

readRenviron(".env")
output_folder=Sys.getenv("OUTPUT_FOLDER")
data_folder=Sys.getenv("DATA_FOLDER")
results_folder=Sys.getenv("RESULTS_FOLDER")


outcome_data <- read_parquet(file.path(results_folder, 'outcome_data_cleaned.parquet'))
antibiotic_data<-read_parquet(file.path(data_folder,"FEVER_COLLATED.parquet")) %>% 
 left_join(outcome_data %>% select(record_id,arm,n_clinicians,hospital_num,hosp_id,clinician_num),by=c("study_id"="record_id")) %>% 
 filter(n_clinicians==1)

antibiotic_data<-antibiotic_data %>% 
mutate(
    anti_correct=case_when(
        antibiotic_pres=="No" & antibiotic_needed == "No" ~ 1.0,
        antibiotic_pres=="No" & antibiotic_needed == "Yes" ~ 0.0,
        antibiotic_pres=="Yes" & necessity %in% c("Clearly justified","Partially justified")  ~ 1.0,    
        antibiotic_pres=="Yes" & necessity == "Not justified" ~ 0.0,
        TRUE ~ NA_real_
    ),    anti_correct2 = if_else(anti_correct == 1.0 & quality == "Poor quality / unsafe" & !is.na(quality), 0.0 , anti_correct)
    )

# glmm with random intercept for clinician and hospital
model_antibiotic=glmer(anti_correct2 ~ arm  + (1|clinician_num) + (1|hospital_num),data=antibiotic_data,family=binomial(link="logit"))
summary(model_antibiotic)   

# gllm for anti_correct
model_antibiotic2=glmer(anti_correct ~ arm  + (1|clinician_num) + (1|hospital_num),data=antibiotic_data,family=binomial(link="logit"))
summary(model_antibiotic2)  

# use tbl_regression and tbl_merge to create a table with the two models
tab1 = tbl_summary(antibiotic_data %>% mutate(anti_correct2b = factor(anti_correct2, levels = c(0, 1), labels = c("Incorrect", "Correct"))),
    include = c("arm"),
    by = "anti_correct2b",
    percent = "row",
    label = list(arm = "Study Arm")
)
tbl1=tbl_regression(model_antibiotic,label = list('arm'="Study Arm",'panel'="Expert Panel"),
    exponentiate = TRUE,
    # tidy_fun = broom.mixed::tidy,
    conf.level = 0.95) %>%
    modify_header(label = "**Variable**") #%>%    modify_caption("**Table X. Odds Ratios for correct antibiotic prescription**")

merge1=tbl_merge(tbls = list(tab1, tbl1),
    tab_spanner = c("**Antibiotic Use**","**Multilevel logistic regression including quality of prescription**")
)
   
# tbl1 %>%
#     as_gt() %>%
#     gtsave(filename = file.path(output_folder, "antibiotic_correct_quality.docx"))

tab2 = tbl_summary(antibiotic_data %>% mutate(anti_correctb = factor(anti_correct, levels = c(0, 1), labels = c("Incorrect", "Correct"))),
    include = c("arm"),
    by = "anti_correctb",
    percent = "row",
    label = list(arm = "Study Arm")
)

tbl2=tbl_regression(model_antibiotic2,label = list('arm'="Study Arm",'panel'="Expert Panel"),
    exponentiate = TRUE,
    # tidy_fun = broom.mixed::tidy,
    conf.level = 0.95) %>%
    modify_header(label = "**Variable**") #%>%    modify_caption("**Table X. Odds Ratios for correct antibiotic prescription**")

merge2=tbl_merge(tbls = list(tab2, tbl2),
    tab_spanner = c("**Antibiotic Use**","**Multilevel logistic regression**"))


tbl_merge(list(merge2,merge1),tab_spanner  = c("**Excluding prescription quality**","**Including prescription quality**")) %>%
   as_gt() %>%
   gtsave(filename = file.path(output_folder, "antibiotic_correct_combined.html")   )

# tbl2 %>%
#     as_gt() %>%
#     gtsave(filename = file.path(output_folder, "antibiotic_correct.html"))  

# tbl_merge (tbls = list(tbl1, tbl2), tab_spanner = c("Including quality of prescription", "Excluding quality of prescription")) %>%
#    as_gt() %>%
#    gtsave(filename = file.path(output_folder, "antibiotic_correct_combined.html"))




for(arm in c(NA,"Control","Intervention")){
    if(!is.na(arm)){
        sankey_data=antibiotic_data %>% filter(arm==!!arm)
    }else{
        sankey_data=antibiotic_data
    }

    sankey_data=sankey_data %>% 
    mutate(antibiotic_pres=recode(antibiotic_pres,"No"="No Antibiotics Prescribed",
    "Yes"="Antibiotics Prescribed"),
    necessity=recode(necessity,"Not justified"="Antibiotic not justified",
    "Partially justified"="Antibiotic partially justified","Clearly justified"="Antibiotic clearly justified"),
    antibiotic_needed=recode(antibiotic_needed,
    "No"="Antibiotic not needed",
    "Yes"="Antibiotic needed")) 

    sankey_vars_doc=c("antibiotic_pres","necessity","antibiotic_needed","quality")
    sankey_levels=list("antibiotic_pres"=list("necessity","antibiotic_needed"),"necessity"=list("quality"))
    sankey_levels2=list()
    counter=1
    for(i in names(sankey_levels)){
    for(j in sankey_levels[[i]]){
        sankey_levels2[[counter]]=list(i,j)
        counter=counter+1
    }
    }

    nodes_doc=list()
    node_conter=0
    for(v in sankey_vars_doc){
    unique_var=levels(sankey_data[[v]])
    if(is.null(unique_var)){
        unique_var=unique(sankey_data[[v]]) %>% na.omit()
    }
    node_id_var=node_conter:(node_conter+length(unique_var)-1)
    nodes_doc[[v]] <- data.frame(id=node_id_var, name=unique_var)
    node_conter=max(node_id_var)+1
    }


    links_doc=list()
    counter=1
    for(i in 1:length(sankey_levels2)){
    var1=sankey_levels2[[i]][[1]]
    var2=sankey_levels2[[i]][[2]]

    tab <- sankey_data %>%
        group_by(!!sym(var1), !!sym(var2)) %>%
        summarise(n = n(), .groups = 'drop') %>% drop_na()
    map_var1=list()
    for(j in seq_len(nrow(nodes_doc[[var1]]))){
        map_var1[[as.character(nodes_doc[[var1]][j,"name"])]]=nodes_doc[[var1]][j,"id"]
    }
    map_var2=list()
    for(j in 1:nrow(nodes_doc[[var2]])){
        map_var2[[as.character(nodes_doc[[var2]][j,"name"])]]=nodes_doc[[var2]][j,"id"]
    }

    tab2 <- tab  %>% 
    mutate(!!sym(var1) := recode(!!sym(var1), !!!map_var1),
            !!sym(var2) := recode(!!sym(var2), !!!map_var2),'var'=paste(var1)) %>% 
            rename(source=!!sym(var1),target=!!sym(var2),value=n)
    links_doc[[counter]] <- tab2 %>% 
        select(source, target, value,var)
        counter=counter+1

    }

    nodes_data=bind_rows(nodes_doc,.id="var") %>% arrange(id) %>% mutate(colorvar=as.character(id)) |>as.data.frame()
    links_data=bind_rows(links_doc) %>% left_join(nodes_data %>% select(id,name,colorvar),by=c("source"="id"))|>as.data.frame()
    links_data2=links_data %>% left_join(nodes_data %>% select(id,name),by=c("target"="id"),suffix=c("_source","_target"))

    sankey = sankeyNetwork(
    Links = links_data,
    Nodes = nodes_data,
    Source = "source",
    Target = "target",
    Value = "value",
    NodeID = "name",
    NodeGroup = "colorvar",
    LinkGroup = "colorvar",
    sinksRight = FALSE,
    fontSize = 14,
    nodeWidth = 20,
    colourScale = JS("d3.scaleOrdinal(d3.schemeCategory20);")
    )

    htmlwidgets::saveWidget(sankey, file.path(output_folder,"Sankey diagram of antibiotic prescription safety.html"))
    webshot2::webshot(file.path(output_folder,"Sankey diagram of antibiotic prescription safety.html"),
     file.path(output_folder,sprintf("Sankey diagram of antibiotic prescription safety - %s.pdf",arm)))

}
