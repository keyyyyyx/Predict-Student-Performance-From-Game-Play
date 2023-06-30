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
rm(q1_0s, new_dat)

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
### Prepare validation data for prediction

val_ids <- val_ids[(val_ids %in% cp1_val$session_id) & 
                     (val_ids %in% cp2_val$session_id) & 
                     (val_ids %in% cp3_val$session_id)]
cp1_val <- cp1_val %>% filter(session_id %in% val_ids)
cp2_val <- cp2_val %>% filter(session_id %in% val_ids)
cp3_val <- cp3_val %>% filter(session_id %in% val_ids)

## Checkpoint 1
cp1_val <- cp1_val %>% select(-c(q1, q2, q3, correctness))

# question 1
q1_test <- cp1_val 

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

# fit using tuned parameters; imbalanced data
xgb_train_q1 <- xgb.DMatrix(data = as.matrix(q1[, -c(1, 28)]), label = (as.numeric(q1$q1)))
q1_test <- left_join(q1_test, labels[(labels$question == 1) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")
xgb_test <- xgb.DMatrix(data = as.matrix(q1_test[, -c(1, 28)]), label = (as.numeric(q1_test$correct)))
xgb_params <- list(
  booster = "gbtree",
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
set.seed(579)
q1_gb <- xgb.train(params = xgb_params, data = xgb_train_q1, nrounds = 5000, verbose = 1, maximize = TRUE)
q1_out_gb <- predict(q1_gb, as.matrix(q1_test[, -c(1, 28)]))
confusionMatrix(as.factor(q1_out_gb), as.factor(q1_test$correct)) # acc = 0.7361
mean(q1_out_gb == q1_test$correct) #  0.736055
auc(q1_out_gb, q1_test$correct) #  0.6699



