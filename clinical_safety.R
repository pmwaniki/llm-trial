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

safety_data<-read_parquet(file.path(data_folder,"CLINICAL_SAFETY_COLLATED.parquet")) 
safety_data=safety_data %>% 
mutate(assess_advice=factor(assess_advice,
levels=c( 'Definitely Safe and Appropriate','Mostly Safe and Appropriate',
 'Neutral (Not Unsafe/Not Inappropriate)', 'Somewhat Unsafe or Inappropriate', 'Unsafe and Inappropriate'),ordered = T))


safety_data=safety_data %>% 
mutate(adherence=factor(adherence,
levels=c('Yes',"Partially","NO"),
labels=c("Yes","Partially","No"),
ordered = T))

safety_data=safety_data %>% 
mutate(justification=factor(justification,
levels=c('Yes','NO'),
labels=c('Yes','No'),
ordered = T))


sankey_vars_doc=c("assess_advice","adherence","justification")

nodes_doc=list()
node_conter=0
for(v in sankey_vars_doc){
  unique_var=levels(safety_data[[v]])
  if(is.null(unique_var)){
    unique_var=unique(safety_data[[v]]) %>% na.omit()
  }
  node_id_var=node_conter:(node_conter+length(unique_var)-1)
  nodes_doc[[v]] <- data.frame(id=node_id_var, name=unique_var)
  node_conter=max(node_id_var)+1
}


links_doc=list()

for(i in 2:length(sankey_vars_doc)){
  var1=sankey_vars_doc[i-1]
  var2=sankey_vars_doc[i]
  
  tab <- safety_data %>%
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
         !!sym(var2) := recode(!!sym(var2), !!!map_var2)) %>% 
         rename(source=!!sym(var1),target=!!sym(var2),value=n)
  links_doc[[var1]] <- tab2 %>% 
    select(source, target, value)

}

nodes_data=bind_rows(nodes_doc,.id="var") %>% arrange(id) %>% mutate(colorvar=as.character(id)) |>as.data.frame()
links_data=bind_rows(links_doc,.id="var") %>% left_join(nodes_data %>% select(id,name,colorvar),by=c("source"="id"))|>as.data.frame()
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

htmlwidgets::saveWidget(sankey, file.path(output_folder,"Sankey diagram of clinical safety.html"))
webshot2::webshot(file.path(output_folder,"Sankey diagram of clinical safety.html"), file.path(output_folder,"Sankey diagram of clinical safety.pdf"),vwidth = 1200, vheight = 800)


tbl_summary(safety_data,
            include=c("assess_advice","adherence","justification"),
            label=list(assess_advice="Assessment of Advice Safety",
                       adherence="Adherence to Advice",
                       justification="Justification of clinician action"),
                       type = list(justification ~ "categorical")) %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(output_folder, "table_clinical_safety.html")   )