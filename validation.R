### -----------------------------------------------------------
### Load packages
library(tidyverse)
library(randomForest)
library(caret)
library(pROC)

### -----------------------------------------------------------
### Load data sets
cp1_transformed <- read.csv("./data/cp1_transformed.csv", colClasses = c("character", rep("numeric", 30)))
cp2_transformed <- read.csv("./data/cp2_transformed.csv", colClasses = c("character", rep("numeric", 46)))
cp3_transformed <- read.csv("./data/cp3_transformed.csv", colClasses = c("character", rep("numeric", 57)))
labels <- read.csv("data/train_labels.csv")
# fix id
labels$question <- substring(labels$session_id, 20)
labels$session_id <- substring(labels$session_id, 1, 17)

### -----------------------------------------------------------
### Prepare data for each question

# Split data into training and validation sets
ids <- unique(labels$session_id)
set.seed(579)
val_ids <- sample(ids, 5000)
# checkpoint 1
cp1_train  <- cp1_transformed %>% filter(!(session_id %in% val_ids))
cp1_val <- cp1_transformed %>% filter(session_id %in% val_ids)
# checkpoint 2
cp2_train  <- cp2_transformed %>% filter(!(session_id %in% val_ids))
cp2_val <- cp2_transformed %>% filter(session_id %in% val_ids)
# checkpoint 3
cp3_train  <- cp3_transformed %>% filter(!(session_id %in% val_ids))
cp3_val <- cp3_transformed %>% filter(session_id %in% val_ids)

## Checkpoint 1

# question 1
q1 <- cp1_train %>% 
  select(-c(q2, q3, correctness))
q1$q1 <- as.factor(q1$q1)
table(q1$q1)
# oversampling imbalanced data
q1_0s <- q1[q1$q1 == 0, ]
set.seed(579)
new_dat <- q1_0s[sample(nrow(q1_0s), 8000, replace = TRUE), ]
q1_balanced <- rbind(q1, new_dat)
table(q1_balanced$q1)
# remove used objects
rm(q1, q1_0s, new_dat)

# question 2
q2 <- cp1_train %>% 
  mutate(correctness  = q1) %>% 
  select(-c(q1, q3))
q2$q2 <- as.factor(q2$q2)
table(q2$q2)
# oversampling imbalanced data
q2_0s <- q2[q2$q2 == 0, ]
set.seed(579)
new_dat <- q2_0s[sample(nrow(q2_0s), 17000, replace = TRUE), ]
q2_balanced <- rbind(q2, new_dat)
table(q2_balanced$q2)
# remove used objects
rm(q2, q2_0s, new_dat)

# question 3
q3 <- cp1_train %>% 
  mutate(correctness  = (q1 + q2)/2) %>% 
  select(-c(q1, q2))
q3$q3 <- as.factor(q3$q3)
table(q3$q3)
# oversampling imbalanced data
q3_0s <- q3[q3$q3 == 0, ]
set.seed(579)
new_dat <- q3_0s[sample(nrow(q3_0s), 16000, replace = TRUE), ]
q3_balanced <- rbind(q3, new_dat)
table(q3_balanced$q3)
# remove used objects
rm(q3, q3_0s, new_dat)

## Checkpoint 2

# question 4
q4 <- cp2_train %>% 
  mutate(correctness  = cp1) %>% 
  select(-c(q1, q2, q3, q5, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q4$q4 <- as.factor(q4$q4)
table(q4$q4)
# oversampling imbalanced data
q4_0s <- q4[q4$q4 == 0, ]
set.seed(579)
new_dat <- q4_0s[sample(nrow(q4_0s), 11000, replace = TRUE), ]
q4_balanced <- rbind(q4, new_dat)
table(q4_balanced$q4)
# remove used objects
rm(q4, q4_0s, new_dat)

# question 5
q5 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + q4)/4) %>% 
  select(-c(q1, q2, q3, q4, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q5$q5 <- as.factor(q5$q5)
table(q5$q5) # original data is balanced
q5_balanced <- q5
# remove used objects
rm(q5)

# question 6
q6 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5)/5) %>% 
  select(-c(q1, q2, q3, q4, q5, q7, q8, q9, q10, q11, q12, q13, cp1))
q6$q6 <- as.factor(q6$q6)
table(q6$q6)
# oversampling imbalanced data
q6_0s <- q6[q6$q6 == 0, ]
set.seed(579)
new_dat <- q6_0s[sample(nrow(q6_0s), 10000, replace = TRUE), ]
q6_balanced <- rbind(q6, new_dat)
table(q6_balanced$q6)
# remove used objects
rm(q6, q6_0s, new_dat)

# question 7
q7 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5 + q6)/6) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q8, q9, q10, q11, q12, q13, cp1))
q7$q7 <- as.factor(q7$q7)
table(q7$q7)
# oversampling imbalanced data
q7_0s <- q7[q7$q7 == 0, ]
set.seed(579)
new_dat <- q7_0s[sample(nrow(q7_0s), 9000, replace = TRUE), ]
q7_balanced <- rbind(q7, new_dat)
table(q7_balanced$q7)
# remove used objects
rm(q7, q7_0s, new_dat)

# question 8
q8 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5 + q6 + q7)/7) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q9, q10, q11, q12, q13, cp1))
q8$q8 <- as.factor(q8$q8)
table(q8$q8)
# oversampling imbalanced data
q8_0s <- q8[q8$q8 == 0, ]
set.seed(579)
new_dat <- q8_0s[sample(nrow(q8_0s), 4000, replace = TRUE), ]
q8_balanced <- rbind(q8, new_dat)
table(q8_balanced$q8)
# remove used objects
rm(q8, q8_0s, new_dat)

# question 9
q9 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10 - q9)/8) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q10, q11, q12, q13, cp1))
q9$q9 <- as.factor(q9$q9)
table(q9$q9)
# oversampling imbalanced data
q9_0s <- q9[q9$q9 == 0, ]
set.seed(579)
new_dat <- q9_0s[sample(nrow(q9_0s), 9000, replace = TRUE), ]
q9_balanced <- rbind(q9, new_dat)
table(q9_balanced$q9)
# remove used objects
rm(q9, q9_0s, new_dat)

# question 10
q10 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10)/9) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q11, q12, q13, cp1))
q10$q10 <- as.factor(q10$q10)
table(q10$q10)  # already balanced
q10_balanced <- q10
# remove used objects
rm(q10)

# question 11
q11 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11)/10) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q12, q13, cp1))
q11$q11 <- as.factor(q11$q11)
table(q11$q11)
# oversampling imbalanced data
q11_0s <- q11[q11$q11 == 0, ]
set.seed(579)
new_dat <- q11_0s[sample(nrow(q11_0s), 4500, replace = TRUE), ]
q11_balanced <- rbind(q11, new_dat)
table(q11_balanced$q11)
# remove used objects
rm(q11, q11_0s, new_dat)

# question 12
q12 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12)/11) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q13, cp1))
q12$q12 <- as.factor(q12$q12)
table(q12$q12)
# oversampling imbalanced data
q12_0s <- q12[q12$q12 == 0, ]
set.seed(579)
new_dat <- q12_0s[sample(nrow(q12_0s), 13000, replace = TRUE), ]
q12_balanced <- rbind(q12, new_dat)
table(q12_balanced$q12)
# remove used objects
rm(q12, q12_0s, new_dat)

# question 13
q13 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13)/12) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, cp1))
q13$q13 <- as.factor(q13$q13)
table(q13$q13)
# oversampling imbalanced data
q13_1s <- q13[q13$q13 == 1, ]
set.seed(579)
new_dat <- q13_1s[sample(nrow(q13_1s), 8000, replace = TRUE), ]
q13_balanced <- rbind(q13, new_dat)
table(q13_balanced$q13)
# remove used objects
rm(q13, q13_1s, new_dat)

## Checkpoint 3

# question 14
q14 <- cp3_train %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10)/13) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q15, 
            q16, q17, q18, cp1, cp2))
q14$q14 <- as.factor(q14$q14)
table(q14$q14)
# oversampling imbalanced data
q14_0s <- q14[q14$q14 == 0, ]
set.seed(579)
new_dat <- q14_0s[sample(nrow(q14_0s), 8000, replace = TRUE), ]
q14_balanced <- rbind(q14, new_dat)
table(q14_balanced$q14)
# remove used objects
rm(q14, q14_0s, new_dat)

# question 15
q15 <- cp3_train %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + q14)/14) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q16, q17, q18, cp1, cp2))
q15$q15 <- as.factor(q15$q15)
table(q15$q15) # already balanced
q15_balanced <- q15
# remove used objects
rm(q15)

# question 16
q16 <- cp3_train %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18 - q17 - q16)/15) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q17, q18, cp1, cp2))
q16$q16 <- as.factor(q16$q16)
table(q16$q16)
# oversampling imbalanced data
q16_0s <- q16[q16$q16 == 0, ]
set.seed(579)
new_dat <- q16_0s[sample(nrow(q16_0s), 9000, replace = TRUE), ]
q16_balanced <- rbind(q16, new_dat)
table(q16_balanced$q16)
# remove used objects
rm(q16, q16_0s, new_dat)

# question 17
q17 <- cp3_train %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18 - q17)/16) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q16, q18, cp1, cp2))
q17$q17 <- as.factor(q17$q17)
table(q17$q17)
# oversampling imbalanced data
q17_0s <- q17[q17$q17 == 0, ]
set.seed(579)
new_dat <- q17_0s[sample(nrow(q17_0s), 7000, replace = TRUE), ]
q17_balanced <- rbind(q17, new_dat)
table(q17_balanced$q17)
# remove used objects
rm(q17, q17_0s, new_dat)

# question 18
q18 <- cp3_train %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18)/17) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q16, q17, cp1, cp2))
q18$q18 <- as.factor(q18$q18)
table(q18$q18)
# oversampling imbalanced data
q18_0s <- q18[q18$q18 == 0, ]
set.seed(579)
new_dat <- q18_0s[sample(nrow(q18_0s), 16000, replace = TRUE), ]
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
### Prepare validation data for prediction

val_ids <- val_ids[(val_ids %in% cp1_val$session_id) & 
                     (val_ids %in% cp2_val$session_id) & 
                     (val_ids %in% cp3_val$session_id)]
cp1_val <- cp1_val %>% filter(session_id %in% val_ids)
cp2_val <- cp2_val %>% filter(session_id %in% val_ids)
cp3_val <- cp3_val %>% filter(session_id %in% val_ids)

### -----------------------------------------------------------
### Predict

# find optimal threshold 
n <- length(val_ids)
results <- as.data.frame(matrix(nrow = 18 * n, ncol = 3)) 
colnames(results) <- c("session_id", "correct", "question")
results[, 3] <- rep(1:18, each = n)

## Checkpoint 1
cp1_val <- cp1_val %>% select(-c(q1, q2, q3, correctness))

# question 1
q1_test <- cp1_val 
results[1:n, 1] <- q1_test$session_id
results[1:n, 2] <- predict(q1_rf, q1_test) %>% as.numeric() - 1

# question 2
q2_test <- cp1_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q2_test <- left_join(q2_test, correctness, by = "session_id")
results[(n+1):(2*n), 1] <- q2_test$session_id
results[(n+1):(2*n), 2] <- predict(q2_rf, q2_test) %>% as.numeric() - 1

# question 3
q3_test <- cp1_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q3_test <- left_join(q3_test, correctness, by = "session_id")
results[(2*n+1):(3*n), 1] <- q3_test$session_id
results[(2*n+1):(3*n), 2] <- predict(q3_rf, q3_test) %>% as.numeric() - 1

## Checkpoint 2
cp2_val <- cp2_val %>% select(-c(q1, q2, q3, cp1, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, correctness))

# question 4
q4_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q4_test <- left_join(q4_test, correctness, by = "session_id")
results[(3*n+1):(4*n), 1] <- q4_test$session_id
results[(3*n+1):(4*n), 2] <- predict(q4_rf, q4_test) %>% as.numeric() - 1

# question 5
q5_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q5_test <- left_join(q5_test, correctness, by = "session_id")
results[(4*n+1):(5*n), 1] <- q5_test$session_id
results[(4*n+1):(5*n), 2] <- predict(q5_rf, q5_test) %>% as.numeric() - 1

# question 6
q6_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q6_test <- left_join(q6_test, correctness, by = "session_id")
results[(5*n+1):(6*n), 1] <- q6_test$session_id
results[(5*n+1):(6*n), 2] <- predict(q6_rf, q6_test) %>% as.numeric() - 1

# question 7
q7_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q7_test <- left_join(q7_test, correctness, by = "session_id")
results[(6*n+1):(7*n), 1] <- q7_test$session_id
results[(6*n+1):(7*n), 2] <- predict(q7_rf, q7_test) %>% as.numeric() - 1

# question 8
q8_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q8_test <- left_join(q8_test, correctness, by = "session_id")
results[(7*n+1):(8*n), 1] <- q8_test$session_id
results[(7*n+1):(8*n), 2] <- predict(q8_rf, q8_test) %>% as.numeric() - 1

# question 9
q9_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q9_test <- left_join(q9_test, correctness, by = "session_id")
results[(8*n+1):(9*n), 1] <- q9_test$session_id
results[(8*n+1):(9*n), 2] <- predict(q9_rf, q9_test) %>% as.numeric() - 1

# question 10
q10_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q10_test <- left_join(q10_test, correctness, by = "session_id")
results[(9*n+1):(10*n), 1] <- q10_test$session_id
results[(9*n+1):(10*n), 2] <- predict(q10_rf, q10_test) %>% as.numeric() - 1

# question 11
q11_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q11_test <- left_join(q11_test, correctness, by = "session_id")
results[(10*n+1):(11*n), 1] <- q11_test$session_id
results[(10*n+1):(11*n), 2] <- predict(q11_rf, q11_test) %>% as.numeric() - 1

# question 12
q12_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q12_test <- left_join(q12_test, correctness, by = "session_id")
results[(11*n+1):(12*n), 1] <- q12_test$session_id
results[(11*n+1):(12*n), 2] <- predict(q12_rf, q12_test) %>% as.numeric() - 1

# question 13
q13_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q13_test <- left_join(q13_test, correctness, by = "session_id")
results[(12*n+1):(13*n), 1] <- q13_test$session_id
results[(12*n+1):(13*n), 2] <- predict(q13_rf, q13_test) %>% as.numeric() - 1

# Checkpoint 3

cp3_val <- cp3_val %>% 
  select(-c(q1, q2, q3, cp1, q4, q5, q6, q7, q8, q9, q10, 
            q11, q12, q13, cp2, q14, q15, q16, q17, q18, correctness))

# question 14
q14_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q14_test <- left_join(q14_test, correctness, by = "session_id")
results[(13*n+1):(14*n), 1] <- q14_test$session_id
results[(13*n+1):(14*n), 2] <- predict(q14_rf, q14_test) %>% as.numeric() - 1

# question 15
q15_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q15_test <- left_join(q15_test, correctness, by = "session_id")
results[(14*n+1):(15*n), 1] <- q15_test$session_id
results[(14*n+1):(15*n), 2] <- predict(q15_rf, q15_test) %>% as.numeric() - 1

# question 16
q16_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q16_test <- left_join(q16_test, correctness, by = "session_id")
results[(15*n+1):(16*n), 1] <- q16_test$session_id
results[(15*n+1):(16*n), 2] <- predict(q16_rf, q16_test) %>% as.numeric() - 1

# question 17
q17_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q17_test <- left_join(q17_test, correctness, by = "session_id")
results[(16*n+1):(17*n), 1] <- q17_test$session_id
results[(16*n+1):(17*n), 2] <- predict(q17_rf, q17_test) %>% as.numeric() - 1

# question 18
q18_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q18_test <- left_join(q18_test, correctness, by = "session_id")
results[(17*n+1):(18*n), 1] <- q18_test$session_id
results[(17*n+1):(18*n), 2] <- predict(q18_rf, q18_test) %>% as.numeric() - 1

### -----------------------------------------------------------
### Compute prediction accuracy

labels_val <- labels %>% filter(session_id %in% val_ids)
colnames(labels_val) <- c("session_id", "label", "question")
labels_val$question <- labels_val$question %>% as.numeric()
results <- left_join(results, labels_val, by = c("session_id", "question"))
# compute prediction accuracy of each session
accuracy <- results %>% 
  mutate(accurate = (correct == label) + 0) %>% 
  group_by(session_id) %>% 
  summarize(accuracy = mean(accurate))
# plot histogram of prediction accuracy by session
hist(accuracy$accuracy)
mean(accuracy$accuracy)
# plot accuracy by question
acc_by_question <- results %>% 
  mutate(accurate = (correct == label) + 0) %>% 
  group_by(question) %>% 
  summarize(accuracy = mean(accurate))
png(filename = "img/PredAccByQuest.png")
ggplot(data = acc_by_question, mapping = aes(x = as.factor(question), y = accuracy)) + 
  geom_col() +
  ggtitle("Barplot of Prediction Accuracy by Question") +
  xlab("question")
dev.off()

### -----------------------------------------------------------
### Compute AUC by question
auc_by_question <- results %>% 
  group_by(question) %>% 
  summarize(auc = auc(label, correct))
# q1
auc(results[results$question ==1, "correct"], results[results$question == 1, "label"])
# overall auc
auc(results$correct, results$label) # 0.6977
plot(roc(results$correct, results$label))

### -----------------------------------------------------------
### Test other models for better accuracy on question 1 

# logistic regression
q1_lr <- glm(q1 ~ ., family = binomial, data = q1_balanced[, -1])
summary(q1_lr)
q1_out <- (predict(q1_lr, q1_test[, -1], type = "response") > 0.5) + 0 
mean(q1_out == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.65; lower than rf

# random forest variants
# original
q1_out_rf <- predict(q1_rf, q1_test) %>% as.numeric() - 1
mean(q1_out_rf == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7225937; rf
# with scaled data, mtry = 5
q1_rf2 <- randomForest(q1 ~ ., data = q1_scaled[, -1], ntree = 500)
q1_out_rf2 <- predict(q1_rf2, scale(q1_test[, -1])) %>% as.numeric() - 1
mean(q1_out_rf2 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7179331; lower than rf
# add number of trees
q1_rf3 <- randomForest(q1 ~ ., data = q1_scaled[, -1], ntree = 1000)
q1_out_rf3 <- predict(q1_rf3, scale(q1_test[, -1])) %>% as.numeric() - 1
mean(q1_out_rf3 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7173; lower than rf2
# mtry = 6
q1_rf4 <- randomForest(q1 ~ ., data = q1_scaled[, -1], ntree = 500, mtry = 15)
q1_out_rf4 <- predict(q1_rf4, scale(q1_test[, -1])) %>% as.numeric() - 1
mean(rep(1, 4935) == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 
# with unbalanced data
q1_rf5 <- randomForest(q1 ~ ., data = q1[, -1], ntree = 500)
q1_out_rf5 <- predict(q1_rf5, q1_test[, -1]) %>% as.numeric() - 1
mean(q1_out_rf5 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7242148;
# with scaled unbalanced data
q1_scaled_ub <- q1
q1_scaled_ub[, -c(1, 28)] <- scale(q1[, -c(1, 28)])
q1_rf6 <- randomForest(q1 ~ ., data = q1_scaled_ub[, -1], ntree = 500)
q1_out_rf6 <- predict(q1_rf6, scale(q1_test[, -1])) %>% as.numeric() - 1
mean(q1_out_rf6 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.725228; highest so far * 
# with scaled unbalanced data; change parameters
q1_rf7 <- randomForest(q1 ~ ., data = q1_scaled_ub[, -1], ntree = 1000)
q1_out_rf7 <- predict(q1_rf7, scale(q1_test[, -1])) %>% as.numeric() - 1
mean(q1_out_rf7 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7258359; no significant improve
# with scaled unbalanced data; change parameters
q1_rf8 <- randomForest(q1 ~ ., data = q1_scaled_ub[, -1], ntree = 500, mtry = 7)
q1_out_rf8 <- predict(q1_rf8, scale(q1_test[, -1])) %>% as.numeric() - 1
mean(q1_out_rf8 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7248227; lower.


### -----------------------------------------------------------
### Other sketches

# SVM with polynomial kernel
q1_scaled <- q1_balanced
q1_scaled[, -c(1, 28)] <- scale(q1_balanced[, -c(1, 28)])
# Fitting SVM to the Training set
library(e1071)
q1_svm = svm(formula = q1 ~ . - session_id,
                 data = q1_scaled,
                 type = 'nu-classification',
                 kernel = 'polynomial')
q1_out_svm <- predict(q1_svm, q1_test)
mean(q1_out_svm == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.717; higher than rf; super slow
# svm with radial kernel
q1_svm2 = svm(formula = q1 ~ . - session_id,
             data = q1_scaled,
             type = 'nu-classification',
             kernel = 'radial')
q1_out_svm2 <- predict(q1_svm2, q1_test)
mean(q1_out_svm2 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7244; higher than rf; super slow
# SVM with C-classigication
q1_svm3 = svm(formula = q1 ~ . - session_id,
             data = q1_scaled,
             type = 'C-classification',
             kernel = 'polynomial')
q1_out_svm3 <- predict(q1_svm3, q1_test)
mean(q1_out_svm3 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.275; much lower than rf; 
# svm with sigmoid kernel
q1_svm4 = svm(formula = q1 ~ . - session_id,
              data = q1_scaled,
              type = 'nu-classification',
              kernel = 'sigmoid')
q1_out_svm4 <- predict(q1_svm4, q1_test)
mean(q1_out_svm4 == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.7244; higher than rf; same as radial kernel

# gradient boosting
library(xgboost)
library(mlr)
# question 1
# balanced data with customized parameters
xgb_train_q1 <- xgb.DMatrix(data = as.matrix(q1_balanced[, -c(1, 28)]), label = (as.numeric(q1_balanced$q1) - 1))
q1_test <- left_join(q1_test, labels[(labels$question == 1) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")
xgb_test <- xgb.DMatrix(data = as.matrix(q1_test[, -c(1, 28)]), label = (as.numeric(q1_test$correct) - 1))
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 10,
  gamma = 4,
  max_delta_step = 1,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "binary:hinge",
  eval_metric = "auc"
)
# compute the best nrounds
q1_xgbcv <- xgb.cv( params = xgb_params, data = xgb_train_q1, nrounds = 5000, nfold = 5, showsd = T, stratified = T, print_every_n = 100, maximize = F, metrics = "error")
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_auc_mean)
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_error_mean)
which.min(q1_xgbcv$evaluation_log$test_error_mean) # 4965
which.max(q1_xgbcv$evaluation_log$test_auc_mean) # 4965
q1_gb <- xgb.train(params = xgb_params, data = xgb_train_q1, nrounds = 4965, verbose = 1)
q1_out_gb <- predict(q1_gb, as.matrix(q1_test[, -c(1, 28)]))
confusionMatrix(as.factor(q1_out_gb), as.factor(q1_test$correct)) # acc = 0.7051     
mean(q1_out_gb == q1_test$correct) # 0.7051334; lower than rf
auc(q1_out_gb, q1_test$correct) # 0.5848

# try default params
xgb_params <- list(
  booster = "gbtree",
  eta = 0.3,
  max_depth = 6,
  gamma = 0,
  max_delta_step = 1,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "binary:hinge",
  eval_metric = "auc"
)
# compute the best nrounds
q1_xgbcv <- xgb.cv( params = xgb_params, data = xgb_train_q1, nrounds = 5000, nfold = 5, showsd = T, stratified = T, print_every_n = 100, maximize = F, metrics = "error")
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_auc_mean)
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_error_mean)
which.min(q1_xgbcv$evaluation_log$test_error_mean) # 3630
which.max(q1_xgbcv$evaluation_log$test_auc_mean) # 3630
q1_gb <- xgb.train(params = xgb_params, data = xgb_train_q1, nrounds = 3630, verbose = 1)
q1_out_gb <- predict(q1_gb, as.matrix(q1_test[, -c(1, 28)]))
confusionMatrix(as.factor(q1_out_gb), as.factor(q1_test$correct)) # acc = 0.6605
mean(q1_out_gb == q1_test$correct) # 0.6604689; lower than rf
auc(q1_out_gb, q1_test$correct) # 0.5615

# try imbalanced data ** best 
xgb_train_q1 <- xgb.DMatrix(data = as.matrix(q1[, -c(1, 28)]), label = (as.numeric(q1$q1) - 1))
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 10,
  gamma = 4,
  max_delta_step = 1,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "binary:hinge",
  eval_metric = "auc"
)
# compute the best nrounds
q1_xgbcv <- xgb.cv( params = xgb_params, data = xgb_train_q1, nrounds = 5000, nfold = 5, showsd = T, stratified = T, print_every_n = 100, maximize = F, metrics = "error")
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_auc_mean)
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_error_mean)
which.min(q1_xgbcv$evaluation_log$test_error_mean) # 356
which.max(q1_xgbcv$evaluation_log$test_auc_mean) # 4839
q1_gb <- xgb.train(params = xgb_params, data = xgb_train_q1, nrounds = 356, verbose = 1)
q1_out_gb <- predict(q1_gb, as.matrix(q1_test[, -c(1, 28)]))
confusionMatrix(as.factor(q1_out_gb), as.factor(q1_test$correct)) # acc = 0.7364592
mean(q1_out_gb == q1_test$correct) # 0.7364592; 
auc(q1_out_gb, q1_test$correct) # 0.673

# optimize other parameters
# create tasks
traintask <- makeClassifTask(data = q1[, -1], target = "q1")
testtask <- makeClassifTask(data = q1_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:hinge", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
# set parameter space
params_try <- makeParamSet(makeIntegerParam("max_depth",lower = 3L,upper = 15L), 
                           makeNumericParam("subsample",lower = 0.5,upper = 1),
                           makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
                           makeNumericParam("eta",lower = 0.01,upper = 0.5),
                           makeNumericParam("gamma",lower = 0,upper = 10),
                           makeIntegerParam("max_delta_step",lower = 1L,upper = 10L))
# set resampling strategy
rdesc <- makeResampleDesc("CV", stratify = T, iters = 5L)
# search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)
# set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
mytune$y 
# [Tune] Result: max_depth=5; subsample=0.822; colsample_bytree=0.536; eta=0.111; gamma=9.57; max_delta_step=9 : acc.test.mean=0.7318900
# [Tune] Result: max_depth=6; subsample=0.655; colsample_bytree=0.615; eta=0.0331; gamma=9.38; max_delta_step=6 : acc.test.mean=0.7328081
# fit using tune result
# try imbalanced data ** best 
xgb_train_q1 <- xgb.DMatrix(data = as.matrix(q1[, -c(1, 28)]), label = (as.numeric(q1$q1) - 1))
xgb_params <- list(
  booster = "gblinear",
  eta = 0.0331,
  max_depth = 6,
  gamma = 9.38,
  max_delta_step = 6,
  subsample = 0.655,
  colsample_bytree = 0.615,
  objective = "binary:hinge",
  eval_metric = "auc"
)
# compute the best nrounds
q1_xgbcv <- xgb.cv(params = xgb_params, data = xgb_train_q1, nrounds = 1000, nfold = 5, showsd = T, stratified = T, print_every_n = 100, maximize = T, metrics = "auc")
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_auc_mean)
plot(q1_xgbcv$evaluation_log$iter, q1_xgbcv$evaluation_log$test_error_mean)
which.min(q1_xgbcv$evaluation_log$test_error_mean) # 356
which.max(q1_xgbcv$evaluation_log$test_auc_mean) # 4839
q1_gb <- xgb.train(params = xgb_params, data = xgb_train_q1, nrounds = 59, verbose = 1)
q1_out_gb <- predict(q1_gb, as.matrix(q1_test[, -c(1, 28)]))
confusionMatrix(as.factor(q1_out_gb), as.factor(q1_test$correct)) # acc = 0.7314
mean(q1_out_gb == q1_test$correct) # 00.7314
auc(q1_out_gb, q1_test$correct) #  0.6243

# Boosted logistic regression
q1_lb <- train(
  q1 ~ ., data = q1_balanced[, -1], method = "LogitBoost",
  trControl = trainControl("cv", number = 10)
)
q1_out_lb <- predict(q1_lb, q1_test[, -1])
mean(q1_out_lb == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.667; lower than rf
# with scaled data
q1_lbs <- train(
  q1 ~ ., data = q1_scaled[, -1], method = "LogitBoost",
  trControl = trainControl("cv", number = 10)
)
q1_out_lbs <- predict(q1_lbs, scale(q1_test[, -1]))
mean(q1_out_lbs == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.6567376; lower than rf

# boosted classification trees
q1_bct <- train(
  q1 ~ ., data = q1_balanced[, -1], method = "ada",
  trControl = trainControl("cv", number = 10)
)
q1_out_bct <- predict(q1_bct, q1_test[, -1])
mean(q1_out_bct == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.6391084; lower than rf
# with scaled data
q1_bcts <- train(
  q1 ~ ., data = q1_scaled[, -1], method = "ada",
  trControl = trainControl("cv", number = 10)
)
q1_out_bcts <- predict(q1_bcts, scale(q1_test[, -1]))
mean(q1_out_bcts == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.6257345; lower than rf

# Bayesian Generalized Linear Model
q1_bglm <- train(
  q1 ~ ., data = q1_balanced[, -1], method = "bayesglm",
  trControl = trainControl("cv", number = 10)
)
q1_out_bglm <- predict(q1_bglm, q1_test[, -1])
mean(q1_out_bglm == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.6494428; lower than rf

# Naive Bayes
q1_nb <- train(
  q1 ~ ., data = q1_balanced[, -1], method = "naive_bayes",
  trControl = trainControl("cv", number = 10)
)
q1_out_nb <- predict(q1_nb, q1_test[, -1])
mean(q1_out_nb == labels[(labels$question == 1) & (labels$session_id %in% val_ids), "correct"]) # 0.6666667; lower than rf

# try meanshift
# try dimension reduction with t-SNE
library(Rtsne)
# convert data into matrix
q1_matrix <- as.matrix(q1_balanced[, -c(1, 28)])
set.seed(527)
tsne_out <- Rtsne(unique(q1_matrix))
q1_tsne <- data.frame(x = tsne_out$Y[,1],
                       y = tsne_out$Y[,2])
# Plotting the 2D visualization of data created by t-SNE
ggplot2::ggplot(q1_tsne, mapping = aes(x = x, y = y)) +
  geom_point() +
  theme_bw()
# choose bandwidth
library(ks)
set.seed(527)
H <- Hpi.diag(q1_tsne)
H
# cluster using mean shift, with selected parameters
library(meanShiftR)
q1_ms <- meanShift(as.matrix(q1_tsne), bandwidth = c(H[1, 1], H[2, 2]), iterations = 300)
unique(q1_ms$assignment) # 22 clusters

# test on q2
# original *
q2_out_rf <- predict(q2_rf, q2_test) %>% as.numeric() - 1
mean(q2_out_rf == labels[(labels$question == 2) & (labels$session_id %in% val_ids), "correct"]) # 0.9805471
# with scaled data, default mtry 
q2_scaled <- q2_balanced
q2_scaled[, -c(1, 29)] <- scale(q2_balanced[, -c(1, 29)])
q2_rf2 <- randomForest(q2 ~ ., data = q2_scaled[, -1], ntree = 500)
q2_out_rf2 <- predict(q2_rf2, scale(q2_test[, -1])) %>% as.numeric() - 1
mean(q2_out_rf2 == labels[(labels$question == 2) & (labels$session_id %in% val_ids), "correct"]) # 0.9805471; similar to rf
# with unbalanced data
q2_rf3 <- randomForest(q2 ~ ., data = q2[, -1], ntree = 500)
q2_out_rf3 <- predict(q2_rf3, q2_test[, -1]) %>% as.numeric() - 1
mean(q2_out_rf3 == labels[(labels$question == 2) & (labels$session_id %in% val_ids), "correct"]) # 0.9805471;
# with scaled unbalanced data
q2_scaled_ub <- q2
q2_scaled_ub[, -c(1, 29)] <- scale(q2[, -c(1, 29)])
q2_rf4 <- randomForest(q2 ~ ., data = q2_scaled_ub[, -1], ntree = 500)
q2_out_rf4 <- predict(q2_rf4, scale(q2_test[, -1])) %>% as.numeric() - 1
mean(q2_out_rf4 == labels[(labels$question == 2) & (labels$session_id %in% val_ids), "correct"]) # 0.9805471; same

# test on q3
# original *
q3_out_rf <- predict(q3_rf, q3_test) %>% as.numeric() - 1
mean(q3_out_rf == labels[(labels$question == 3) & (labels$session_id %in% val_ids), "correct"]) # 0.9381966
# with scaled data, default mtry 
q3_scaled <- q3_balanced
q3_scaled[, -c(1, 29)] <- scale(q3_balanced[, -c(1, 29)])
q3_rf2 <- randomForest(q3 ~ ., data = q3_scaled[, -1], ntree = 500)
q3_out_rf2 <- predict(q3_rf2, scale(q3_test[, -1])) %>% as.numeric() - 1
mean(q3_out_rf2 == labels[(labels$question == 3) & (labels$session_id %in% val_ids), "correct"]) # 0.9375887; lower than  rf
# with unbalanced data
q3_rf3 <- randomForest(q3 ~ ., data = q3[, -1], ntree = 500)
q3_out_rf3 <- predict(q3_rf3, q3_test[, -1]) %>% as.numeric() - 1
mean(q3_out_rf3 == labels[(labels$question == 3) & (labels$session_id %in% val_ids), "correct"]) # 0.9383992; slightly higher
# with scaled unbalanced data
q3_scaled_ub <- q3
q3_scaled_ub[, -c(1, 29)] <- scale(q3[, -c(1, 29)])
q3_rf4 <- randomForest(q3 ~ ., data = q3_scaled_ub[, -1], ntree = 500)
q3_out_rf4 <- predict(q3_rf4, scale(q3_test[, -1])) %>% as.numeric() - 1
mean(q3_out_rf4 == labels[(labels$question == 3) & (labels$session_id %in% val_ids), "correct"]) # 0.9381966; same

# test on q5
# original *
q5_out_rf <- predict(q5_rf, q5_test) %>% as.numeric() - 1
mean(q5_out_rf == labels[(labels$question == 5) & (labels$session_id %in% val_ids), "correct"]) # 0.6050659
# with scaled data, default mtry 
q5_scaled <- q5_balanced
q5_scaled[, -c(1, 34)] <- scale(q5_balanced[, -c(1, 34)])
q5_rf2 <- randomForest(q5 ~ ., data = q5_scaled[, -1], ntree = 500)
q5_out_rf2 <- predict(q5_rf2, scale(q5_test[, -1])) %>% as.numeric() - 1
mean(q5_out_rf2 == labels[(labels$question == 5) & (labels$session_id %in% val_ids), "correct"]) # 0.5967579; lower than  rf

### --------------------------------------------
### find optimal threshold 

threshold <- seq(0.3, 0.6, by = 0.01) 
acc <- numeric(length = length(threshold))

for (i in 1:length(threshold)) {
thres <- threshold[i]

n <- length(val_ids)
results <- as.data.frame(matrix(nrow = 18 * n, ncol = 3)) 
colnames(results) <- c("session_id", "correct", "question")
results[, 3] <- rep(1:18, each = n)

## Checkpoint 1

# question 1
q1_test <- cp1_val 
results[1:n, 1] <- q1_test$session_id
results[1:n, 2] <- ((predict(q1_rf, q1_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 2
q2_test <- cp1_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) %>% 
  as.data.frame()
q2_test <- left_join(q2_test, correctness, by = "session_id")
results[(n+1):(2*n), 1] <- q2_test$session_id
results[(n+1):(2*n), 2] <- ((predict(q2_rf, q2_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 3
q3_test <- cp1_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q3_test <- left_join(q3_test, correctness, by = "session_id")
results[(2*n+1):(3*n), 1] <- q3_test$session_id
results[(2*n+1):(3*n), 2] <- ((predict(q3_rf, q3_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

## Checkpoint 2

# question 4
q4_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q4_test <- left_join(q4_test, correctness, by = "session_id")
results[(3*n+1):(4*n), 1] <- q4_test$session_id
results[(3*n+1):(4*n), 2] <- ((predict(q4_rf, q4_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 5
q5_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q5_test <- left_join(q5_test, correctness, by = "session_id")
results[(4*n+1):(5*n), 1] <- q5_test$session_id
results[(4*n+1):(5*n), 2] <- ((predict(q5_rf, q5_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 6
q6_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q6_test <- left_join(q6_test, correctness, by = "session_id")
results[(5*n+1):(6*n), 1] <- q6_test$session_id
results[(5*n+1):(6*n), 2] <- ((predict(q6_rf, q6_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 7
q7_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q7_test <- left_join(q7_test, correctness, by = "session_id")
results[(6*n+1):(7*n), 1] <- q7_test$session_id
results[(6*n+1):(7*n), 2] <- ((predict(q7_rf, q7_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 8
q8_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q8_test <- left_join(q8_test, correctness, by = "session_id")
results[(7*n+1):(8*n), 1] <- q8_test$session_id
results[(7*n+1):(8*n), 2] <- ((predict(q8_rf, q8_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 9
q9_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q9_test <- left_join(q9_test, correctness, by = "session_id")
results[(8*n+1):(9*n), 1] <- q9_test$session_id
results[(8*n+1):(9*n), 2] <- ((predict(q9_rf, q9_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 10
q10_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q10_test <- left_join(q10_test, correctness, by = "session_id")
results[(9*n+1):(10*n), 1] <- q10_test$session_id
results[(9*n+1):(10*n), 2] <- ((predict(q10_rf, q10_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 11
q11_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q11_test <- left_join(q11_test, correctness, by = "session_id")
results[(10*n+1):(11*n), 1] <- q11_test$session_id
results[(10*n+1):(11*n), 2] <- ((predict(q11_rf, q11_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 12
q12_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q12_test <- left_join(q12_test, correctness, by = "session_id")
results[(11*n+1):(12*n), 1] <- q12_test$session_id
results[(11*n+1):(12*n), 2] <- ((predict(q12_rf, q12_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 13
q13_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q13_test <- left_join(q13_test, correctness, by = "session_id")
results[(12*n+1):(13*n), 1] <- q13_test$session_id
results[(12*n+1):(13*n), 2] <- ((predict(q13_rf, q13_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# Checkpoint 3

# question 14
q14_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q14_test <- left_join(q14_test, correctness, by = "session_id")
results[(13*n+1):(14*n), 1] <- q14_test$session_id
results[(13*n+1):(14*n), 2] <- ((predict(q14_rf, q14_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 15
q15_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q15_test <- left_join(q15_test, correctness, by = "session_id")
results[(14*n+1):(15*n), 1] <- q15_test$session_id
results[(14*n+1):(15*n), 2] <- ((predict(q15_rf, q15_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 16
q16_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q16_test <- left_join(q16_test, correctness, by = "session_id")
results[(15*n+1):(16*n), 1] <- q16_test$session_id
results[(15*n+1):(16*n), 2] <- ((predict(q16_rf, q16_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 17
q17_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q17_test <- left_join(q17_test, correctness, by = "session_id")
results[(16*n+1):(17*n), 1] <- q17_test$session_id
results[(16*n+1):(17*n), 2] <- ((predict(q17_rf, q17_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

# question 18
q18_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q18_test <- left_join(q18_test, correctness, by = "session_id")
results[(17*n+1):(18*n), 1] <- q18_test$session_id
results[(17*n+1):(18*n), 2] <- ((predict(q18_rf, q18_test, type = "prob")[, "1"] > thres) + 0) %>% as.vector()

### -----------------------------------------------------------
### Compute prediction accuracy

labels_val <- labels %>% filter(session_id %in% val_ids)
colnames(labels_val) <- c("session_id", "label", "question")
labels_val$question <- labels_val$question %>% as.numeric()
results <- left_join(results, labels_val, by = c("session_id", "question"))
acc[i] <- mean(results$correct == results$label)
}

plot(x = threshold, y = acc, type = "l")
threshold[which.max(acc)]
