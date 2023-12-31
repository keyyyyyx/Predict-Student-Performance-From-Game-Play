### -----------------------------------------------------------
### Load packages
library(tidyverse)
library(randomForest)

### -----------------------------------------------------------
### Load data sets
cp1_transformed <- read.csv("./data/cp1_transformed.csv", colClasses = c("character", rep("numeric", 30)))
cp2_transformed <- read.csv("./data/cp2_transformed.csv", colClasses = c("character", rep("numeric", 46)))
cp3_transformed <- read.csv("./data/cp3_transformed.csv", colClasses = c("character", rep("numeric", 57)))

### -----------------------------------------------------------
### Prepare data for each question

## Checkpoint 1

# question 1
q1 <- cp1_transformed %>% 
  select(-c(q2, q3, correctness))
q1$q1 <- as.factor(q1$q1)
table(q1$q1)
# oversampling imbalanced data
q1_0s <- q1[q1$q1 == 0, ]
set.seed(579)
new_dat <- q1_0s[sample(nrow(q1_0s), 10000, replace = TRUE), ]
q1_balanced <- rbind(q1, new_dat)
table(q1_balanced$q1)
# remove used objects
rm(q1, q1_0s, new_dat)

# question 2
q2 <- cp1_transformed %>% 
  mutate(correctness  = q1) %>% 
  select(-c(q1, q3))
q2$q2 <- as.factor(q2$q2)
table(q2$q2)
# oversampling imbalanced data
q2_0s <- q2[q2$q2 == 0, ]
set.seed(579)
new_dat <- q2_0s[sample(nrow(q2_0s), 20000, replace = TRUE), ]
q2_balanced <- rbind(q2, new_dat)
table(q2_balanced$q2)
# remove used objects
rm(q2, q2_0s, new_dat)

# question 3
q3 <- cp1_transformed %>% 
  mutate(correctness  = (q1 + q2)/2) %>% 
  select(-c(q1, q2))
q3$q3 <- as.factor(q3$q3)
table(q3$q3)
# oversampling imbalanced data
q3_0s <- q3[q3$q3 == 0, ]
set.seed(579)
new_dat <- q3_0s[sample(nrow(q3_0s), 20000, replace = TRUE), ]
q3_balanced <- rbind(q3, new_dat)
table(q3_balanced$q3)
# remove used objects
rm(q3, q3_0s, new_dat)

## Checkpoint 2

# question 4
q4 <- cp2_transformed %>% 
  mutate(correctness  = cp1) %>% 
  select(-c(q1, q2, q3, q5, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q4$q4 <- as.factor(q4$q4)
table(q4$q4)
# oversampling imbalanced data
q4_0s <- q4[q4$q4 == 0, ]
set.seed(579)
new_dat <- q4_0s[sample(nrow(q4_0s), 13000, replace = TRUE), ]
q4_balanced <- rbind(q4, new_dat)
table(q4_balanced$q4)
# remove used objects
rm(q4, q4_0s, new_dat)

# question 5
q5 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4)/4) %>% 
  select(-c(q1, q2, q3, q4, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q5$q5 <- as.factor(q5$q5)
table(q5$q5) # original data is balanced
q5_balanced <- q5
# remove used objects
rm(q5)

# question 6
q6 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5)/5) %>% 
  select(-c(q1, q2, q3, q4, q5, q7, q8, q9, q10, q11, q12, q13, cp1))
q6$q6 <- as.factor(q6$q6)
table(q6$q6)
# oversampling imbalanced data
q6_0s <- q6[q6$q6 == 0, ]
set.seed(579)
new_dat <- q6_0s[sample(nrow(q6_0s), 13000, replace = TRUE), ]
q6_balanced <- rbind(q6, new_dat)
table(q6_balanced$q6)
# remove used objects
rm(q6, q6_0s, new_dat)

# question 7
q7 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5 + q6)/6) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q8, q9, q10, q11, q12, q13, cp1))
q7$q7 <- as.factor(q7$q7)
table(q7$q7)
# oversampling imbalanced data
q7_0s <- q7[q7$q7 == 0, ]
set.seed(579)
new_dat <- q7_0s[sample(nrow(q7_0s), 10000, replace = TRUE), ]
q7_balanced <- rbind(q7, new_dat)
table(q7_balanced$q7)
# remove used objects
rm(q7, q7_0s, new_dat)

# question 8
q8 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5 + q6 + q7)/7) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q9, q10, q11, q12, q13, cp1))
q8$q8 <- as.factor(q8$q8)
table(q8$q8)
# oversampling imbalanced data
q8_0s <- q8[q8$q8 == 0, ]
set.seed(579)
new_dat <- q8_0s[sample(nrow(q8_0s), 6000, replace = TRUE), ]
q8_balanced <- rbind(q8, new_dat)
table(q8_balanced$q8)
# remove used objects
rm(q8, q8_0s, new_dat)

# question 9
q9 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10 - q9)/8) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q10, q11, q12, q13, cp1))
q9$q9 <- as.factor(q9$q9)
table(q9$q9)
# oversampling imbalanced data
q9_0s <- q9[q9$q9 == 0, ]
set.seed(579)
new_dat <- q9_0s[sample(nrow(q9_0s), 10000, replace = TRUE), ]
q9_balanced <- rbind(q9, new_dat)
table(q9_balanced$q9)
# remove used objects
rm(q9, q9_0s, new_dat)

# question 10
q10 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10)/9) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q11, q12, q13, cp1))
q10$q10 <- as.factor(q10$q10)
table(q10$q10)  # already balanced
q10_balanced <- q10
# remove used objects
rm(q10)

# question 11
q11 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11)/10) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q12, q13, cp1))
q11$q11 <- as.factor(q11$q11)
table(q11$q11)
# oversampling imbalanced data
q11_0s <- q11[q11$q11 == 0, ]
set.seed(579)
new_dat <- q11_0s[sample(nrow(q11_0s), 8000, replace = TRUE), ]
q11_balanced <- rbind(q11, new_dat)
table(q11_balanced$q11)
# remove used objects
rm(q11, q11_0s, new_dat)

# question 12
q12 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12)/11) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q13, cp1))
q12$q12 <- as.factor(q12$q12)
table(q12$q12)
# oversampling imbalanced data
q12_0s <- q12[q12$q12 == 0, ]
set.seed(579)
new_dat <- q12_0s[sample(nrow(q12_0s), 15000, replace = TRUE), ]
q12_balanced <- rbind(q12, new_dat)
table(q12_balanced$q12)
# remove used objects
rm(q12, q12_0s, new_dat)

# question 13
q13 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13)/12) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, cp1))
q13$q13 <- as.factor(q13$q13)
table(q13$q13)
# oversampling imbalanced data
q13_1s <- q13[q13$q13 == 1, ]
set.seed(579)
new_dat <- q13_1s[sample(nrow(q13_1s), 10000, replace = TRUE), ]
q13_balanced <- rbind(q13, new_dat)
table(q13_balanced$q13)
# remove used objects
rm(q13, q13_1s, new_dat)

## Checkpoint 3

# question 14
q14 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10)/13) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q15, 
            q16, q17, q18, cp1, cp2))
q14$q14 <- as.factor(q14$q14)
table(q14$q14)
# oversampling imbalanced data
q14_0s <- q14[q14$q14 == 0, ]
set.seed(579)
new_dat <- q14_0s[sample(nrow(q14_0s), 10000, replace = TRUE), ]
q14_balanced <- rbind(q14, new_dat)
table(q14_balanced$q14)
# remove used objects
rm(q14, q14_0s, new_dat)

# question 15
q15 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + q14)/14) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q16, q17, q18, cp1, cp2))
q15$q15 <- as.factor(q15$q15)
table(q15$q15) # already balanced
q15_balanced <- q15
# remove used objects
rm(q15)

# question 16
q16 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18 - q17 - q16)/15) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q17, q18, cp1, cp2))
q16$q16 <- as.factor(q16$q16)
table(q16$q16)
# oversampling imbalanced data
q16_0s <- q16[q16$q16 == 0, ]
set.seed(579)
new_dat <- q16_0s[sample(nrow(q16_0s), 10000, replace = TRUE), ]
q16_balanced <- rbind(q16, new_dat)
table(q16_balanced$q16)
# remove used objects
rm(q16, q16_0s, new_dat)

# question 17
q17 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18 - q17)/16) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q16, q18, cp1, cp2))
q17$q17 <- as.factor(q17$q17)
table(q17$q17)
# oversampling imbalanced data
q17_0s <- q17[q17$q17 == 0, ]
set.seed(579)
new_dat <- q17_0s[sample(nrow(q17_0s), 10000, replace = TRUE), ]
q17_balanced <- rbind(q17, new_dat)
table(q17_balanced$q17)
# remove used objects
rm(q17, q17_0s, new_dat)

# question 18
q18 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18)/17) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q16, q17, cp1, cp2))
q18$q18 <- as.factor(q18$q18)
table(q18$q18)
# oversampling imbalanced data
q18_0s <- q18[q18$q18 == 0, ]
set.seed(579)
new_dat <- q18_0s[sample(nrow(q18_0s), 20000, replace = TRUE), ]
q18_balanced <- rbind(q18, new_dat)
table(q18_balanced$q18)
# remove used objects
rm(q18, q18_0s, new_dat)

### -----------------------------------------------------------
### Train models
# question 1
q1_rf <- randomForest(q1 ~ . - session_id, data = q1_balanced, ntree = 500)
# rm(q1_balanced)
# question 2
q2_rf <- randomForest(q2 ~ . - session_id, data = q2_balanced, ntree = 500)
# rm(q2_balanced)
# question 3
q3_rf <- randomForest(q3 ~ . - session_id, data = q3_balanced, ntree = 500)
# rm(q3_balanced)
# question 4
q4_rf <- randomForest(q4 ~ . - session_id, data = q4_balanced, ntree = 500)
# rm(q4_balanced)
# question 5
q5_rf <- randomForest(q5 ~ . - session_id, data = q5_balanced, ntree = 500)
# rm(q5_balanced)
# question 6
q6_rf <- randomForest(q6 ~ . - session_id, data = q6_balanced, ntree = 500)
# rm(q6_balanced)
# question 7
q7_rf <- randomForest(q7 ~ . - session_id, data = q7_balanced, ntree = 500)
# rm(q7_balanced)
# question 8
q8_rf <- randomForest(q8 ~ . - session_id, data = q8_balanced, ntree = 500)
# rm(q8_balanced)
# question 9
q9_rf <- randomForest(q9 ~ . - session_id, data = q9_balanced, ntree = 500)
# rm(q9_balanced)
# question 10
q10_rf <- randomForest(q10 ~ . - session_id, data = q10_balanced, ntree = 500)
# rm(q10_balanced)
# question 11
q11_rf <- randomForest(q11 ~ . - session_id, data = q11_balanced, ntree = 500)
# rm(q11_balanced)
# question 12
q12_rf <- randomForest(q12 ~ . - session_id, data = q12_balanced, ntree = 500)
# rm(q12_balanced)
# question 13
q13_rf <- randomForest(q13 ~ . - session_id, data = q13_balanced, ntree = 500)
# rm(q13_balanced)
# question 14
q14_rf <- randomForest(q14 ~ . - session_id, data = q14_balanced, ntree = 500)
# rm(q14_balanced)
# question 15
q15_rf <- randomForest(q15 ~ . - session_id, data = q15_balanced, ntree = 500)
# rm(q15_balanced)
# question 16
q16_rf <- randomForest(q16 ~ . - session_id, data = q16_balanced, ntree = 500)
# rm(q16_balanced)
# question 17
q17_rf <- randomForest(q17 ~ . - session_id, data = q17_balanced, ntree = 500)
# rm(q17_balanced)
# question 18
q18_rf <- randomForest(q18 ~ . - session_id, data = q18_balanced, ntree = 500)
# rm(q18_balanced)

### -----------------------------------------------------------
### Prepare testing data for prediction

# read data
test <- read.csv("data/test.csv", 
                 colClasses = c("character", "character", "numeric", "character", "character",
                                "numeric", "numeric", "numeric",  "numeric",  "numeric",  
                                "numeric", "numeric", "character", "character", "character", 
                                "character", "numeric", "numeric", "numeric", "character"))
# replace NAs in hover duration with 0
test$hover_duration[is.na(test$hover_duration)] <- 0
# replace NAs in page with -1
test$page[is.na(test$page)] <- -1

# group data by session id and levels, summarize covariates
test_grouped <- test %>% 
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

# split dataset by checkpoint
test_cp1 <- test_grouped %>% filter(level_group == "0-4")
test_cp2 <- test_grouped %>% filter(level_group == "5-12")
# checkpoint2$cp1 <- checkpoint1$correctness 
test_cp3 <- test_grouped %>% filter(level_group == "13-22")
# checkpoint3$cp1 <- checkpoint1$correctness
# checkpoint3$cp2 <- checkpoint2$correctness

# drop NA columns
test_cp1 <- test_cp1 %>% select_if(~!any(is.na(.)))
test_cp2 <- test_cp2 %>% select_if(~!any(is.na(.)))
test_cp3 <- test_cp3 %>% select_if(~!any(is.na(.)))

# drop unuseful columns
col_rm <- colnames(test_cp1)[!(colnames(test_cp1) %in% colnames(q1_balanced))]
test_cp1[, col_rm] <- NULL
col_rm <- colnames(test_cp2)[!(colnames(test_cp2) %in% colnames(q4_balanced))]
test_cp2[, col_rm] <- NULL
col_rm <- colnames(test_cp3)[!(colnames(test_cp3) %in% colnames(q14_balanced))]
test_cp3[, col_rm] <- NULL
rm(col_rm)

### -----------------------------------------------------------
### Predict
n <- nrow(test_cp1)
results <- as.data.frame(matrix(nrow = 18 * n, ncol = 3)) 
colnames(results) <- c("session_id", "correct", "question")
results[, 3] <- rep(1:18, each = n)

## Checkpoint 1

# question 1
q1_test <- test_cp1
results[1:n, 1] <- q1_test$session_id
results[1:n, 2] <- predict(q1_rf, q1_test) %>% as.numeric() - 1

# question 2
q2_test <- test_cp1
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q2_test <- left_join(q2_test, correctness, by = "session_id")
results[(n+1):(2*n), 1] <- q2_test$session_id
results[(n+1):(2*n), 2] <- predict(q2_rf, q2_test) %>% as.numeric() - 1

# question 3
q3_test <- test_cp1
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q3_test <- left_join(q3_test, correctness, by = "session_id")
results[(2*n+1):(3*n), 1] <- q3_test$session_id
results[(2*n+1):(3*n), 2] <- predict(q3_rf, q3_test) %>% as.numeric() - 1

## Checkpoint 2

# question 4
q4_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q4_test <- left_join(q4_test, correctness, by = "session_id")
results[(3*n+1):(4*n), 1] <- q4_test$session_id
results[(3*n+1):(4*n), 2] <- predict(q4_rf, q4_test) %>% as.numeric() - 1

# question 5
q5_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q5_test <- left_join(q5_test, correctness, by = "session_id")
results[(4*n+1):(5*n), 1] <- q5_test$session_id
results[(4*n+1):(5*n), 2] <- predict(q5_rf, q5_test) %>% as.numeric() - 1

# question 6
q6_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q6_test <- left_join(q6_test, correctness, by = "session_id")
results[(5*n+1):(6*n), 1] <- q6_test$session_id
results[(5*n+1):(6*n), 2] <- predict(q6_rf, q6_test) %>% as.numeric() - 1

# question 7
q7_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q7_test <- left_join(q7_test, correctness, by = "session_id")
results[(6*n+1):(7*n), 1] <- q7_test$session_id
results[(6*n+1):(7*n), 2] <- predict(q7_rf, q7_test) %>% as.numeric() - 1

# question 8
q8_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q8_test <- left_join(q8_test, correctness, by = "session_id")
results[(7*n+1):(8*n), 1] <- q8_test$session_id
results[(7*n+1):(8*n), 2] <- predict(q8_rf, q8_test) %>% as.numeric() - 1

# question 9
q9_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q9_test <- left_join(q9_test, correctness, by = "session_id")
results[(8*n+1):(9*n), 1] <- q9_test$session_id
results[(8*n+1):(9*n), 2] <- predict(q9_rf, q9_test) %>% as.numeric() - 1

# question 10
q10_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q10_test <- left_join(q10_test, correctness, by = "session_id")
results[(9*n+1):(10*n), 1] <- q10_test$session_id
results[(9*n+1):(10*n), 2] <- predict(q10_rf, q10_test) %>% as.numeric() - 1

# question 11
q11_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q11_test <- left_join(q11_test, correctness, by = "session_id")
results[(10*n+1):(11*n), 1] <- q11_test$session_id
results[(10*n+1):(11*n), 2] <- predict(q11_rf, q11_test) %>% as.numeric() - 1

# question 12
q12_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q12_test <- left_join(q12_test, correctness, by = "session_id")
results[(11*n+1):(12*n), 1] <- q12_test$session_id
results[(11*n+1):(12*n), 2] <- predict(q12_rf, q12_test) %>% as.numeric() - 1

# question 13
q13_test <- test_cp2
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q13_test <- left_join(q13_test, correctness, by = "session_id")
results[(12*n+1):(13*n), 1] <- q13_test$session_id
results[(12*n+1):(13*n), 2] <- predict(q13_rf, q13_test) %>% as.numeric() - 1

# Checkpoint 3

# question 14
q14_test <- test_cp3
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q14_test <- left_join(q14_test, correctness, by = "session_id")
results[(13*n+1):(14*n), 1] <- q14_test$session_id
results[(13*n+1):(14*n), 2] <- predict(q14_rf, q14_test) %>% as.numeric() - 1

# question 15
q15_test <- test_cp3
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q15_test <- left_join(q15_test, correctness, by = "session_id")
results[(14*n+1):(15*n), 1] <- q15_test$session_id
results[(14*n+1):(15*n), 2] <- predict(q15_rf, q15_test) %>% as.numeric() - 1

# question 16
q16_test <- test_cp3
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q16_test <- left_join(q16_test, correctness, by = "session_id")
results[(15*n+1):(16*n), 1] <- q16_test$session_id
results[(15*n+1):(16*n), 2] <- predict(q16_rf, q16_test) %>% as.numeric() - 1

# question 17
q17_test <- test_cp3
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q17_test <- left_join(q17_test, correctness, by = "session_id")
results[(16*n+1):(17*n), 1] <- q17_test$session_id
results[(16*n+1):(17*n), 2] <- predict(q17_rf, q17_test) %>% as.numeric() - 1

# question 18
q18_test <- test_cp3
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q18_test <- left_join(q18_test, correctness, by = "session_id")
results[(17*n+1):(18*n), 1] <- q18_test$session_id
results[(17*n+1):(18*n), 2] <- predict(q18_rf, q18_test) %>% as.numeric() - 1

results$session_id <- paste0(results$session_id, "_q", results$question)
results$question <- NULL







