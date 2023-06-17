
### -----------------------------------------------------------
### Setting up the packages
library(xtable)
library(tidyverse)
library(rigr)
library(glmnet)
library(caret)

### -----------------------------------------------------------
### load datasets
## original datasets availale on Kaggle;
## https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data
labels <- read.csv("data/train_labels.csv")
train <- read.csv("data/train.csv", 
                  colClasses = c("character", "character", "numeric", "character", 
                                 "character", "numeric", "numeric", "numeric",  "numeric",  
                                 "numeric",  "numeric", "numeric", "character", "character", 
                                 "character", "character", "numeric", "numeric", "numeric", 
                                 "character"))
test <- read.csv("data/test.csv", 
                 colClasses = c("character", "character", "numeric", "character", "character",
                                "numeric", "numeric", "numeric",  "numeric",  "numeric",  
                                "numeric", "numeric", "character", "character", "character", 
                                "character", "numeric", "numeric", "numeric", "character"))

# fix id
labels$question <- substring(labels$session_id, 19)
labels$session_id <- substring(labels$session_id, 1, 17)

# replace NAs in hover duration with 0
train$hover_duration[is.na(train$hover_duration)] <- 0
# replace NAs in page with -1
train$page[is.na(train$page)] <- -1

### -----------------------------------------------------------
### Data Cleaning
# group data by session id and levels, summarize covariates
train_grouped <- train %>% 
  group_by(session_id, level_group) %>% 
  summarise(fullscreen = mean(fullscreen), hq = mean(hq), music = mean(music), 
            hover_duration = sum(hover_duration), n_actions = n(), 
            elapsed_time = max(elapsed_time)-min(elapsed_time), 
            n_event_name = length(unique(event_name)), n_name = length(unique(name)), 
            n_fqid = length(unique(fqid)), n_room_fqid = length(unique(room_fqid)), 
            n_text_fqid = length(unique(text_fqid)), n_notebook_click = sum(event_name == "notebook_click"),
            n_object_hover = sum(event_name == "object_hover"), n_map_hover = sum(event_name == "map_hover"),
            n_cutscene_click = sum(event_name == "cutscene_click"), 
            n_person_click = sum(event_name == "person_click"),
            n_navigate_click = sum(event_name == "navigate_click"),
            n_observation_click = sum(event_name == "observation_click"),
            n_notification_click = sum(event_name == "notification_click"),
            n_object_click = sum(event_name == "object_click"),
            n_map_click = sum(event_name == "map_click"),
            n_notebook_click = sum(event_name == "notebook_click"), n_page1 = sum(page == 1), 
            n_page2 = sum(page == 2), n_page3 = sum(page == 3), n_page4 = sum(page == 4), 
            n_page5 = sum(page == 5), n_page6 = sum(page == 6), n_page0 = sum(page == 0),
            n_historicalsociety = sum(grepl("historicalsociety", room_fqid)),
            n_kohlcenter = sum(grepl("kohlcenter", room_fqid)), n_capitol = sum(grepl("capitol", room_fqid)),
            n_humanecology = sum(grepl("humanecology", room_fqid)), n_drycleaner = sum(grepl("drycleaner", room_fqid)),
            n_library = sum(grepl("library", room_fqid)), n_wildlifecenter = sum(grepl("wildlife", room_fqid)),
            n_flaghouse = sum(grepl("flaghouse", room_fqid)))

# categorize levels into levels groups
labels$level_group <- ifelse(labels$question %in% c("q1", "q2", "q3"), "0-4", 
                             ifelse(labels$question %in% c("q14", "q15", "q16", "q17", "q18"), "13-22", "5-12"))
# compute correctness in each session for each level group
labels_grouped <- labels %>% 
  group_by(session_id, level_group) %>% 
  summarise(correctness = mean(correct))
# make a wide label, each column represents a question
labels_wide <- labels %>% pivot_wider(id_cols = c(session_id, level_group), names_from = question, values_from = correct)

# merge datasets
train_merged <- merge(train_grouped, labels_grouped, by = intersect(names(train_grouped), names(labels_grouped)))
train_merged <- merge(train_merged, labels_wide, by = intersect(names(train_merged), names(labels_wide)))
# Export merged training set as csv
write.csv(train_merged, "data/train_merged.csv", row.names=FALSE)

# split dataset by checkpoint
checkpoint1 <- train_merged %>% filter(train_merged$level_group == "0-4")
checkpoint2 <- train_merged %>% filter(train_merged$level_group == "5-12")
checkpoint2$cp1 <- checkpoint1$correctness 
checkpoint3 <- train_merged %>% filter(train_merged$level_group == "13-22")
checkpoint3$cp1 <- checkpoint1$correctness
checkpoint3$cp2 <- checkpoint2$correctness

# drop NA columns
checkpoint1 <- checkpoint1 %>% select_if(~!any(is.na(.)))
checkpoint2 <- checkpoint2 %>% select_if(~!any(is.na(.)))
checkpoint3 <- checkpoint3 %>% select_if(~!any(is.na(.)))

# drop unuseful columns
col_rm <- colnames(checkpoint1)[c(FALSE, TRUE, lapply(checkpoint1[, -c(1, 2)], mean) == 0)]
checkpoint1[, col_rm] <- NULL
col_rm <- colnames(checkpoint2)[c(FALSE, TRUE, lapply(checkpoint2[, -c(1, 2)], mean) == 0)]
checkpoint2[, col_rm] <- NULL
col_rm <- colnames(checkpoint3)[c(FALSE, TRUE, lapply(checkpoint3[, -c(1, 2)], mean) == 0)]
checkpoint3[, col_rm] <- NULL

# add performance of q1-3 to checkpoint2
for (i in 1:3) {
  checkpoint2[, paste0("q", i)] <- checkpoint1[, paste0("q", i)]
  checkpoint3[, paste0("q", i)] <- checkpoint1[, paste0("q", i)]
}
# add performance of q1-13 to checkpoint3 
for (i in 1:13) {
  checkpoint3[, paste0("q", i)] <- checkpoint2[, paste0("q", i)]
}

# Export checkpoint 1, 2, 3 as csv
write.csv(checkpoint1, "data/checkpoint1.csv", row.names=FALSE)
write.csv(checkpoint2, "data/checkpoint2.csv", row.names=FALSE)
write.csv(checkpoint3, "data/checkpoint3.csv", row.names=FALSE)