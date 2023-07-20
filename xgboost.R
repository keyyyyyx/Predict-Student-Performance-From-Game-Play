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
rm(q2_0s, new_dat)

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
rm(q3_0s, new_dat)

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
rm(q4_0s, new_dat)

# question 5
q5 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + q4)/4) %>% 
  select(-c(q1, q2, q3, q4, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q5$q5 <- as.factor(q5$q5)
table(q5$q5) # original data is balanced
q5_balanced <- q5


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
rm(q6_0s, new_dat)

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
rm(q7_0s, new_dat)

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
rm(q8_0s, new_dat)

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
rm(q9_0s, new_dat)

# question 10
q10 <- cp2_train %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10)/9) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q11, q12, q13, cp1))
q10$q10 <- as.factor(q10$q10)
table(q10$q10)  # already balanced
q10_balanced <- q10

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
rm(q11_0s, new_dat)

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
rm(q12_0s, new_dat)

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
rm(q13_1s, new_dat)

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
rm(q14_0s, new_dat)

# question 15
q15 <- cp3_train %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + q14)/14) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q16, q17, q18, cp1, cp2))
q15$q15 <- as.factor(q15$q15)
table(q15$q15) # already balanced
q15_balanced <- q15

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
rm(q16_0s, new_dat)

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
rm(q17_0s, new_dat)

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
rm(q18_0s, new_dat)

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

### -----------------------------------------------------------
### Model selection (based on performance on question 1)
library(xgboost)
library(mlr)

# try balanced data with randomly customized parameters
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

### -----------------------------------------------------------
### Use imbalanced data; optimize other parameters
# create tasks
traintask <- makeClassifTask(data = q1[, -1], target = "q1")
testtask <- makeClassifTask(data = q1_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
# [Tune] Result: max_depth=5; subsample=0.822; colsample_bytree=0.536; eta=0.111; gamma=9.57; max_delta_step=9 : acc.test.mean=0.7318900
# ub hinge: [Tune] Result: max_depth=6; subsample=0.655; colsample_bytree=0.615; eta=0.0331; gamma=9.38; max_delta_step=6 : acc.test.mean=0.7328081
# ub logistic: [Tune] Result: max_depth=15; subsample=0.584; colsample_bytree=0.534; eta=0.0316; gamma=9.72; max_delta_step=7 : acc.test.mean=0.7337795
# ub logloss: [Tune] Result: max_depth=6; subsample=0.963; colsample_bytree=0.764; eta=0.119; gamma=7.33; max_delta_step=4 : acc.test.mean=0.7297853

# fit using tuned parameters; imbalanced data
xgb_train_q1 <- xgb.DMatrix(data = as.matrix(q1[, -c(1, 28)]), label = (as.numeric(q1$q1) - 1))
q1_test <- left_join(q1_test, labels[(labels$question == 1) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")
xgb_test <- xgb.DMatrix(data = as.matrix(q1_test[, -c(1, 28)]), label = (as.numeric(q1_test$correct)))
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0316,
  max_depth = 15,
  gamma = 9.72,
  max_delta_step = 7,
  subsample = 0.584,
  colsample_bytree = 0.534,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.119,
  max_depth = 6,
  gamma = 7.33,
  max_delta_step = 4,
  subsample = 0.963,
  colsample_bytree = 0.764,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 1
set.seed(579)
q1_gb <- xgb.train(params = xgb_params, data = xgb_train_q1, nrounds = 5000, verbose = 1, maximize = TRUE)
# make predictions
q1_out_gb <- predict(q1_gb, as.matrix(q1_test[, -c(1, 28)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q1_out_gb > thres) + 0
  accs[i] = mean(pred == q1_test$correct)
}
plot(thresholds, accs)
q1_pred <- (q1_out_gb > thresholds[which.max(accs)]) + 0
# compute accuracy and auc
CM <- confusionMatrix(as.factor(q1_pred), as.factor(q1_test$correct)) # acc = 0.7361
CM
auc <- auc(q1_pred, q1_test$correct) #  0.6699
auc

# initialize eval df
eval <- as.data.frame(matrix(nrow = 0, ncol = 5))
colnames(eval) <- c("question", "accuracy", "auc", "data", "eval_metric")
# record eval metrics for question 1 
eval <- rbind(eval, cbind(question = 1,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "auc"))

# initialize results df
n <- length(val_ids)
results <- as.data.frame(matrix(nrow = 18 * n, ncol = 3)) 
colnames(results) <- c("session_id", "correct", "question")
results[, 3] <- rep(1:18, each = n)

# add predictions of question 1 to results df
results[1:n, 1] <- q1_test$session_id
results[1:n, 2] <- q1_out_gb

### -----------------------------------------------------------
### question 2
# prepare tesing data
q2_test <- cp1_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q2_test <- left_join(q2_test, correctness, by = "session_id")
q2_test <- left_join(q2_test, labels[(labels$question == 2) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q2[, -1], target = "q2")
testtask <- makeClassifTask(data = q2_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# auc & logloss: [Tune] Result: max_depth=5; subsample=0.608; colsample_bytree=0.971; eta=0.077; gamma=7.53; max_delta_step=1 : acc.test.mean=0.9787326

# fit using tuned parameters; imbalanced data
xgb_train_q2 <- xgb.DMatrix(data = as.matrix(q2[, -c(1, 29)]), label = (as.numeric(q2$q2) - 1))
xgb_params <- list(
  booster = "gbtree",
  eta = 0.077,
  max_depth = 5,
  gamma = 7.53,
  max_delta_step = 1,
  subsample = 0.608,
  colsample_bytree = 0.971,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# train xgboost for question 1
set.seed(579)
q2_gb <- xgb.train(params = xgb_params, data = xgb_train_q2, nrounds = 5000, verbose = 1, maximize = TRUE)
q2_out_gb <- predict(q2_gb, as.matrix(q2_test[, -c(1, 29)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q2_out_gb > thres) + 0
  accs[i] = mean(pred == q2_test$correct)
}
plot(thresholds, accs)
q2_pred <- (q2_out_gb > thresholds[which.max(accs)]) + 0
CM <- confusionMatrix(as.factor(q2_pred), as.factor(q2_test$correct)) # acc = 0.9796  
CM
auc <- auc(q2_pred, q2_test$correct) # 0.9898
auc

# record eval metrics for question 1 
eval <- rbind(eval, cbind(question = 2,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "logloss"))

# add predictions of question 2 to results df
results[(n+1):(2*n), 1] <- q2_test$session_id
results[(n+1):(2*n), 2] <- q2_pred

### -----------------------------------------------------------
### question 3
# prepare tesing data
q3_test <- cp1_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q3_test <- left_join(q3_test, correctness, by = "session_id")
q3_test <- left_join(q3_test, labels[(labels$question == 3) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q3[, -1], target = "q3")
testtask <- makeClassifTask(data = q3_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# auc & logloss: [Tune] Result: max_depth=3; subsample=0.865; colsample_bytree=0.857; eta=0.187; gamma=9.22; max_delta_step=10 : acc.test.mean=0.9341466

# fit using tuned parameters; imbalanced data
xgb_train_q3 <- xgb.DMatrix(data = as.matrix(q3[, -c(1, 29)]), label = (as.numeric(q3$q3) - 1))
xgb_params <- list(
  booster = "gbtree",
  eta = 0.187,
  max_depth = 3,
  gamma = 9.22,
  max_delta_step = 10,
  subsample = 0.865,
  colsample_bytree = 0.857,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# train xgboost for question 1
set.seed(579)
q3_gb <- xgb.train(params = xgb_params, data = xgb_train_q3, nrounds = 5000, verbose = 1, maximize = TRUE)
q3_out_gb <- predict(q3_gb, as.matrix(q3_test[, -c(1, 29)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q3_out_gb > thres) + 0
  accs[i] = mean(pred == q3_test$correct)
}
plot(thresholds, accs)
q3_pred <- (q3_out_gb > thresholds[which.max(accs)]) + 0
CM <- confusionMatrix(as.factor(q3_pred), as.factor(q3_test$correct)) # acc = 0.9317
CM
auc <- auc(q3_pred, q3_test$correct) #  NA
auc

# record eval metrics for question 1 
eval <- rbind(eval, cbind(question = 3,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = NA,
                          data = "unbalanced",
                          eval_metric = "logloss"))

# add predictions of question 3 to results df
results[(2*n+1):(3*n), 1] <- q3_test$session_id
results[(2*n+1):(3*n), 2] <- q3_pred

### -----------------------------------------------------------
### checkpoint 2
cp2_val <- cp2_val %>% select(-c(q1, q2, q3, cp1, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, correctness))

### -----------------------------------------------------------
### question 4
# prepare tesing data
q4_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q4_test <- left_join(q4_test, correctness, by = "session_id")
q4_test <- left_join(q4_test, labels[(labels$question == 4) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q4_balanced[, -1], target = "q4")
testtask <- makeClassifTask(data = q4_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# balanced: [Tune] Result: max_depth=10; subsample=0.525; colsample_bytree=0.578; eta=0.139; gamma=1.35; max_delta_step=3 : acc.test.mean=0.9301553
# balanced logloss: [Tune] Result: max_depth=11; subsample=0.586; colsample_bytree=0.623; eta=0.0764; gamma=0.144; max_delta_step=7 : acc.test.mean=0.9422934
# unbalanced: [Tune] Result: max_depth=3; subsample=0.788; colsample_bytree=0.789; eta=0.169; gamma=6.35; max_delta_step=5 : acc.test.mean=0.8110197

# fit using tuned parameters; imbalanced data
xgb_train_q4 <- xgb.DMatrix(data = as.matrix(q4_balanced[, -c(1, 34)]), label = (as.numeric(q4_balanced$q4)) - 1)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.139,
  max_depth = 10,
  gamma = 1.35,
  max_delta_step = 3,
  subsample = 0.525,
  colsample_bytree = 0.578,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0764,
  max_depth = 11,
  gamma = 0.144,
  max_delta_step = 7,
  subsample = 0.586,
  colsample_bytree = 0.623,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.169,
  max_depth = 3,
  gamma = 6.35,
  max_delta_step = 5,
  subsample = 0.788,
  colsample_bytree = 0.789,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.169,
  max_depth = 3,
  gamma = 6.35,
  max_delta_step = 5,
  subsample = 0.788,
  colsample_bytree = 0.789,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 1
set.seed(579)
q4_gb <- xgb.train(params = xgb_params, data = xgb_train_q4, nrounds = 5000, verbose = 1, maximize = FALSE)
q4_out_gb <- predict(q4_gb, as.matrix(q4_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q4_out_gb > thres) + 0
  accs[i] = mean(pred == q4_test$correct)
}
q4_pred <- (q4_out_gb > thresholds[which.max(accs)]) + 0
CM <- confusionMatrix(as.factor(q4_pred), as.factor(q4_test$correct)) # balanced acc = 0.8003; unbalanced acc = 0.8027  
CM
auc <- auc(q4_pred, q4_test$correct) # balanced auc = 0.8189; unbalanced auc = 0.7322
auc

# record eval metrics for question 4
eval <- rbind(eval, cbind(question = 4,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

eval[7, ] <- cbind(question = 4,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "auc")

# add predictions of question 2 to results df
results[(3*n+1):(4*n), 1] <- q4_test$session_id
results[(3*n+1):(4*n), 2] <- q4_pred

### -----------------------------------------------------------
### question 5
# prepare tesing data
q5_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q5_test <- left_join(q5_test, correctness, by = "session_id")
q5_test <- left_join(q5_test, labels[(labels$question == 5) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q5[, -1], target = "q5")
testtask <- makeClassifTask(data = q5_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# [Tune] Result: max_depth=6; subsample=0.987; colsample_bytree=0.901; eta=0.0636; gamma=8.33; max_delta_step=8 : acc.test.mean=0.6601065
# logloss: [Tune] Result: max_depth=7; subsample=0.775; colsample_bytree=0.577; eta=0.0355; gamma=7.24; max_delta_step=4 : acc.test.mean=0.6567525

# fit using tuned parameters; imbalanced data
xgb_train_q5 <- xgb.DMatrix(data = as.matrix(q5[, -c(1, 34)]), label = (as.numeric(q5$q5)) - 1)
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0636,
  max_depth = 6,
  gamma = 8.33,
  max_delta_step = 8,
  subsample = 0.987,
  colsample_bytree = 0.901,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0355,
  max_depth = 7,
  gamma = 7.24,
  max_delta_step = 4,
  subsample = 0.775,
  colsample_bytree = 0.577,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q5_gb <- xgb.train(params = xgb_params, data = xgb_train_q5, nrounds = 5000, verbose = 1, maximize = FALSE)
q5_out_gb <- predict(q5_gb, as.matrix(q5_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q5_out_gb > thres) + 0
  accs[i] = mean(pred == q5_test$correct)
}
plot(thresholds, accs)
q5_pred <- (q5_out_gb > thresholds[which.max(accs)]) + 0
CM <- confusionMatrix(as.factor(q5_pred), as.factor(q5_test$correct)) # q4 balanced acc = 0.6186; ub acc = 0.619
CM
auc <- auc(q5_pred, q5_test$correct) # q4 balanced auc = 0.6152; ub auc = 0.6156
auc

# record eval metrics for question 4
eval <- rbind(eval, cbind(question = 5,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "logloss"))

# add predictions of question 2 to results df
results[(4*n+1):(5*n), 1] <- q5_test$session_id
results[(4*n+1):(5*n), 2] <- q5_pred

### -----------------------------------------------------------
### question 6
# prepare tesing data
q6_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q6_test <- left_join(q6_test, correctness, by = "session_id")
q6_test <- left_join(q6_test, labels[(labels$question == 6) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q6_balanced[, -1], target = "q6")
testtask <- makeClassifTask(data = q6_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=5; subsample=0.86; colsample_bytree=0.967; eta=0.106; gamma=8.64; max_delta_step=1 : acc.test.mean=0.7897156
# balance: [Tune] Result: max_depth=6; subsample=0.741; colsample_bytree=0.82; eta=0.202; gamma=0.0397; max_delta_step=6 : acc.test.mean=0.9110340
# balanced logloss: [Tune] Result: max_depth=13; subsample=0.923; colsample_bytree=0.667; eta=0.11; gamma=0.59; max_delta_step=6 : acc.test.mean=0.9238784

# fit using tuned parameters; imbalanced data
xgb_train_q6 <- xgb.DMatrix(data = as.matrix(q6_balanced[, -c(1, 34)]), label = (as.numeric(q6_balanced$q6)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.106,
  max_depth = 5,
  gamma = 8.64,
  max_delta_step = 1,
  subsample = 0.86,
  colsample_bytree = 0.967,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.106,
  max_depth = 5,
  gamma = 8.64,
  max_delta_step = 1,
  subsample = 0.86,
  colsample_bytree = 0.967,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.202,
  max_depth = 6,
  gamma = 0.0397,
  max_delta_step = 6,
  subsample = 0.741,
  colsample_bytree = 0.82,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.11,
  max_depth = 13,
  gamma = 0.59,
  max_delta_step = 6,
  subsample = 0.923,
  colsample_bytree = 0.667,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q6_gb <- xgb.train(params = xgb_params, data = xgb_train_q6, nrounds = 5000, verbose = 1, maximize = FALSE)
q6_out_gb <- predict(q6_gb, as.matrix(q6_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q6_out_gb > thres) + 0
  accs[i] = mean(pred == q6_test$correct)
}
plot(thresholds, accs)
q6_pred <- (q6_out_gb > thresholds[which.max(accs)]) + 0
CM <- confusionMatrix(as.factor(q6_pred), as.factor(q6_test$correct)) # balanced acc = 0.7148; ub acc = 0.6977
CM
auc <- auc(q6_pred, q6_test$correct) # balanced auc = 0.5958; ub auc = 0.606
auc

# record eval metrics for question 6
eval <- rbind(eval, cbind(question = 6,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "logloss"))
eval[14, ] <- cbind(question = 6,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "logloss")

# add predictions of question 2 to results df
results[(5*n+1):(6*n), 1] <- q6_test$session_id
results[(5*n+1):(6*n), 2] <- q6_pred

### -----------------------------------------------------------
### question 7
# prepare tesing data
q7_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q7_test <- left_join(q7_test, correctness, by = "session_id")
q7_test <- left_join(q7_test, labels[(labels$question == 7) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q7[, -1], target = "q7")
testtask <- makeClassifTask(data = q7_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced auc & logloss: [Tune] Result: max_depth=3; subsample=0.822; colsample_bytree=0.811; eta=0.0654; gamma=4.53; max_delta_step=8 : acc.test.mean=0.7457020
# balance: [Tune] Result: max_depth=14; subsample=0.715; colsample_bytree=0.819; eta=0.0873; gamma=0.568; max_delta_step=1 : acc.test.mean=0.8855027

# fit using tuned parameters; imbalanced data
xgb_train_q7 <- xgb.DMatrix(data = as.matrix(q7[, -c(1, 34)]), label = (as.numeric(q7$q7)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0654,
  max_depth = 3,
  gamma = 4.53,
  max_delta_step = 8,
  subsample = 0.822,
  colsample_bytree = 0.811,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0654,
  max_depth = 3,
  gamma = 4.53,
  max_delta_step = 8,
  subsample = 0.822,
  colsample_bytree = 0.811,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0873,
  max_depth = 14,
  gamma = 0.568,
  max_delta_step = 1,
  subsample = 0.715,
  colsample_bytree = 0.819,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0873,
  max_depth = 14,
  gamma = 0.568,
  max_delta_step = 1,
  subsample = 0.715,
  colsample_bytree = 0.819,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q7_gb <- xgb.train(params = xgb_params, data = xgb_train_q7, nrounds = 5000, verbose = 1, maximize = FALSE)
q7_out_gb <- predict(q7_gb, as.matrix(q7_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q7_out_gb > thres) + 0
  accs[i] = mean(pred == q7_test$correct)
}
plot(thresholds, accs)
q7_pred <- (q7_out_gb > thresholds[which.max(accs)]) + 0
CM <- confusionMatrix(as.factor(q7_pred), as.factor(q7_test$correct)) # balanced acc = 0.733; ub acc = 0.735
CM
auc <- auc(q7_pred, q7_test$correct) # balanced auc = 0.631; ub auc = 0.6602
auc

# record eval metrics for question 7
eval <- rbind(eval, cbind(question = 7,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 2 to results df
results[(6*n+1):(7*n), 1] <- q7_test$session_id
results[(6*n+1):(7*n), 2] <- q7_pred

### -----------------------------------------------------------
### question 8
# prepare tesing data
q8_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q8_test <- left_join(q8_test, correctness, by = "session_id")
q8_test <- left_join(q8_test, labels[(labels$question == 8) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q8[, -1], target = "q8")
testtask <- makeClassifTask(data = q8_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=3; subsample=0.591; colsample_bytree=0.792; eta=0.0579; gamma=1.68; max_delta_step=5 : acc.test.mean=0.6212289
# unbalanced; logloss: [Tune] Result: max_depth=3; subsample=0.637; colsample_bytree=0.85; eta=0.0407; gamma=5.26; max_delta_step=6 : acc.test.mean=0.6230125
# balance: [Tune] Result: max_depth=7; subsample=0.767; colsample_bytree=0.648; eta=0.177; gamma=1.14; max_delta_step=6 : acc.test.mean=0.7064550
# balanced logloss: [Tune] Result: max_depth=7; subsample=0.767; colsample_bytree=0.648; eta=0.177; gamma=1.14; max_delta_step=6 : acc.test.mean=0.7064550

# fit using tuned parameters; imbalanced data
xgb_train_q8 <- xgb.DMatrix(data = as.matrix(q8[, -c(1, 34)]), label = (as.numeric(q8$q8)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0579,
  max_depth = 3,
  gamma = 1.68,
  max_delta_step = 5,
  subsample = 0.591,
  colsample_bytree = 0.792,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0407,
  max_depth = 3,
  gamma = 5.26,
  max_delta_step = 6,
  subsample = 0.637,
  colsample_bytree = 0.85,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.177,
  max_depth = 7,
  gamma = 1.14,
  max_delta_step = 6,
  subsample = 0.767,
  colsample_bytree = 0.648,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.177,
  max_depth = 7,
  gamma = 1.14,
  max_delta_step = 6,
  subsample = 0.767,
  colsample_bytree = 0.648,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q8_gb <- xgb.train(params = xgb_params, data = xgb_train_q8, nrounds = 5000, verbose = 1, maximize = FALSE)
q8_out_gb <- predict(q8_gb, as.matrix(q8_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q8_out_gb > thres) + 0
  accs[i] = mean(pred == q8_test$correct)
}
plot(thresholds, accs)
q8_pred <- (q8_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q8_pred), as.factor(q8_test$correct)) # balanced acc = 0.6271; ub acc = 0.6251; ub acc logloss: 0.6281
CM
auc <- auc(q8_pred, q8_test$correct) # balanced auc = 0.5974; ub auc = 0.6077; ub auc logloss: 0.6357
auc

# record eval metrics for question 8
eval <- rbind(eval, cbind(question = 8,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 2 to results df
results[(7*n+1):(8*n), 1] <- q8_test$session_id
results[(7*n+1):(8*n), 2] <- q8_pred

### -----------------------------------------------------------
### question 9
# prepare tesing data
q9_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q9_test <- left_join(q9_test, correctness, by = "session_id")
q9_test <- left_join(q9_test, labels[(labels$question == 9) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q9[, -1], target = "q9")
testtask <- makeClassifTask(data = q9_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=3; subsample=0.63; colsample_bytree=0.509; eta=0.105; gamma=9.55; max_delta_step=9 : acc.test.mean=0.7603547
# unbalanced; logloss: [Tune] Result: max_depth=6; subsample=0.81; colsample_bytree=0.747; eta=0.0594; gamma=8.86; max_delta_step=2 : acc.test.mean=0.7560289
# balance: [Tune] Result: max_depth=11; subsample=0.805; colsample_bytree=0.963; eta=0.333; gamma=1.21; max_delta_step=4 : acc.test.mean=0.8792101
# balanced logloss: [Tune] Result: max_depth=14; subsample=0.515; colsample_bytree=0.83; eta=0.14; gamma=1.01; max_delta_step=6 : acc.test.mean=0.8872479

# fit using tuned parameters; imbalanced data
xgb_train_q9 <- xgb.DMatrix(data = as.matrix(q9[, -c(1, 34)]), label = (as.numeric(q9$q9)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.105,
  max_depth = 3,
  gamma = 9.55,
  max_delta_step = 9,
  subsample = 0.63,
  colsample_bytree = 0.509,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0594,
  max_depth = 6,
  gamma = 8.86,
  max_delta_step = 2,
  subsample = 0.81,
  colsample_bytree = 0.747,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.333,
  max_depth = 11,
  gamma = 1.21,
  max_delta_step = 4,
  subsample = 0.805,
  colsample_bytree = 0.963,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.14,
  max_depth = 14,
  gamma = 1.01,
  max_delta_step = 6,
  subsample = 0.515,
  colsample_bytree = 0.83,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q9_gb <- xgb.train(params = xgb_params, data = xgb_train_q9, nrounds = 5000, verbose = 1, maximize = FALSE)
q9_out_gb <- predict(q9_gb, as.matrix(q9_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q9_out_gb > thres) + 0
  accs[i] = mean(pred == q9_test$correct)
}
plot(thresholds, accs)
q9_pred <- (q9_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q9_pred), as.factor(q9_test$correct)) # balanced acc = 0.6271; ub acc = 0.6251; ub acc logloss: 0.6281
CM
auc <- auc(q9_pred, q9_test$correct) # balanced auc = 0.5974; ub auc = 0.6077; ub auc logloss: 0.6357
auc

# record eval metrics for question 8
eval <- rbind(eval, cbind(question = 9,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 2 to results df
results[(8*n+1):(9*n), 1] <- q9_test$session_id
results[(8*n+1):(9*n), 2] <- q9_pred

### -----------------------------------------------------------
### question 10
# prepare tesing data
q10_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q10_test <- left_join(q10_test, correctness, by = "session_id")
q10_test <- left_join(q10_test, labels[(labels$question == 10) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q10[, -1], target = "q10")
testtask <- makeClassifTask(data = q10_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=4; subsample=0.626; colsample_bytree=0.887; eta=0.0367; gamma=2.38; max_delta_step=3 : acc.test.mean=0.6372339
# unbalanced; logloss: [Tune] Result: max_depth=9; subsample=0.91; colsample_bytree=0.696; eta=0.114; gamma=5.93; max_delta_step=2 : acc.test.mean=0.6289068

# fit using tuned parameters; imbalanced data
xgb_train_q10 <- xgb.DMatrix(data = as.matrix(q10[, -c(1, 34)]), label = (as.numeric(q10$q10)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0367,
  max_depth = 4,
  gamma = 2.38,
  max_delta_step = 3,
  subsample = 0.626,
  colsample_bytree = 0.887,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.114,
  max_depth = 9,
  gamma = 5.93,
  max_delta_step = 2,
  subsample = 0.91,
  colsample_bytree = 0.696,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q10_gb <- xgb.train(params = xgb_params, data = xgb_train_q10, nrounds = 5000, verbose = 1, maximize = FALSE)
q10_out_gb <- predict(q10_gb, as.matrix(q10_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q10_out_gb > thres) + 0
  accs[i] = mean(pred == q10_test$correct)
}
plot(thresholds, accs)
q10_pred <- (q10_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q10_pred), as.factor(q10_test$correct)) # balanced acc = 0.6271; ub acc = 0.6251; ub acc logloss: 0.6281
CM
auc <- auc(q10_pred, q10_test$correct) # balanced auc = 0.5974; ub auc = 0.6077; ub auc logloss: 0.6357
auc

# record eval metrics for question 8
eval <- rbind(eval, cbind(question = 10,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "auc"))
eval[30, ] <- cbind(question = 10,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "logloss")

# add predictions of question 2 to results df
results[(9*n+1):(10*n), 1] <- q10_test$session_id
results[(9*n+1):(10*n), 2] <- q10_pred

### -----------------------------------------------------------
### question 11
# prepare tesing data
q11_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q11_test <- left_join(q11_test, correctness, by = "session_id")
q11_test <- left_join(q11_test, labels[(labels$question == 11) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q11[, -1], target = "q11")
testtask <- makeClassifTask(data = q11_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=7; subsample=0.848; colsample_bytree=0.585; eta=0.0327; gamma=6.44; max_delta_step=8 : acc.test.mean=0.6645944
# unbalanced; logloss: [Tune] Result: max_depth=11; subsample=0.965; colsample_bytree=0.957; eta=0.108; gamma=7.57; max_delta_step=1 : acc.test.mean=0.6590791
# balanced: [Tune] Result: max_depth=12; subsample=0.686; colsample_bytree=0.987; eta=0.325; gamma=1.81; max_delta_step=4 : acc.test.mean=0.7512831
# balanced logloss: [Tune] Result: max_depth=11; subsample=0.592; colsample_bytree=0.697; eta=0.484; gamma=0.233; max_delta_step=2 : acc.test.mean=0.7444550

# fit using tuned parameters; imbalanced data
xgb_train_q11 <- xgb.DMatrix(data = as.matrix(q11[, -c(1, 34)]), label = (as.numeric(q11$q11)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0327,
  max_depth = 7,
  gamma = 6.44,
  max_delta_step = 8,
  subsample = 0.848,
  colsample_bytree = 0.585,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.108,
  max_depth = 11,
  gamma = 7.57,
  max_delta_step = 1,
  subsample = 0.965,
  colsample_bytree = 0.967,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.325,
  max_depth = 12,
  gamma = 1.81,
  max_delta_step = 4,
  subsample = 0.686,
  colsample_bytree = 0.987,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.484,
  max_depth = 11,
  gamma = 0.233,
  max_delta_step = 2,
  subsample = 0.592,
  colsample_bytree = 0.697,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q11_gb <- xgb.train(params = xgb_params, data = xgb_train_q11, nrounds = 5000, verbose = 1, maximize = FALSE)
q11_out_gb <- predict(q11_gb, as.matrix(q11_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q11_out_gb > thres) + 0
  accs[i] = mean(pred == q11_test$correct)
}
plot(thresholds, accs)
q11_pred <- (q11_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q11_pred), as.factor(q11_test$correct)) # balanced acc = 0.6271; ub acc = 0.6251; ub acc logloss: 0.6281
CM
auc <- auc(q11_pred, q11_test$correct) # balanced auc = 0.5974; ub auc = 0.6077; ub auc logloss: 0.6357
auc

# record eval metrics for question 8
eval <- rbind(eval, cbind(question = 11,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 2 to results df
results[(10*n+1):(11*n), 1] <- q11_test$session_id
results[(10*n+1):(11*n), 2] <- q11_pred

### -----------------------------------------------------------
### question 12
# prepare tesing data
q12_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q12_test <- left_join(q12_test, correctness, by = "session_id")
q12_test <- left_join(q12_test, labels[(labels$question == 12) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q12[, -1], target = "q12")
testtask <- makeClassifTask(data = q12_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=9; subsample=0.504; colsample_bytree=0.722; eta=0.0294; gamma=8.82; max_delta_step=3 : acc.test.mean=0.8614145
# unbalanced; logloss: [Tune] Result: max_depth=4; subsample=0.925; colsample_bytree=0.544; eta=0.127; gamma=6.85; max_delta_step=1 : acc.test.mean=0.8628202
# [Tune] Result: max_depth=15; subsample=0.556; colsample_bytree=0.641; eta=0.0532; gamma=0.328; max_delta_step=7 : acc.test.mean=0.9778053
# balanced logloss: [Tune] Result: max_depth=12; subsample=0.883; colsample_bytree=0.783; eta=0.477; gamma=0.594; max_delta_step=4 : acc.test.mean=0.9605956

# fit using tuned parameters; imbalanced data
xgb_train_q12 <- xgb.DMatrix(data = as.matrix(q12[, -c(1, 34)]), label = (as.numeric(q12$q12)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0294,
  max_depth = 9,
  gamma = 8.82,
  max_delta_step = 3,
  subsample = 0.504,
  colsample_bytree = 0.722,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.127,
  max_depth = 4,
  gamma = 6.85,
  max_delta_step = 1,
  subsample = 0.925,
  colsample_bytree = 0.544,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0532,
  max_depth = 15,
  gamma = 0.328,
  max_delta_step = 7,
  subsample = 0.556,
  colsample_bytree = 0.641,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.477,
  max_depth = 12,
  gamma = 0.594,
  max_delta_step = 4,
  subsample = 0.883,
  colsample_bytree = 0.783,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q12_gb <- xgb.train(params = xgb_params, data = xgb_train_q12, nrounds = 5000, verbose = 1, maximize = FALSE)
q12_out_gb <- predict(q12_gb, as.matrix(q12_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q12_out_gb > thres) + 0
  accs[i] = mean(pred == q12_test$correct)
}
plot(thresholds, accs)
q12_pred <- (q12_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q12_pred), as.factor(q12_test$correct)) 
CM
auc <- auc(q12_pred, q12_test$correct) 
auc

# record eval metrics for question 8
eval <- rbind(eval, cbind(question = 12,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

eval[36, ] <- cbind(question = 12,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "logloss")

# add predictions of question 2 to results df
results[(11*n+1):(12*n), 1] <- q12_test$session_id
results[(11*n+1):(12*n), 2] <- q12_pred

### -----------------------------------------------------------
### question 13
# prepare tesing data
q13_test <- cp2_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q13_test <- left_join(q13_test, correctness, by = "session_id")
q13_test <- left_join(q13_test, labels[(labels$question == 13) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q13[, -1], target = "q13")
testtask <- makeClassifTask(data = q13_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=4; subsample=0.626; colsample_bytree=0.887; eta=0.0367; gamma=2.38; max_delta_step=3 : acc.test.mean=0.7334815
# unbalanced; logloss: [Tune] Result: max_depth=4; subsample=0.704; colsample_bytree=0.854; eta=0.0366; gamma=0.806; max_delta_step=7 : acc.test.mean=0.7337512
# [Tune] [Tune] Result: max_depth=15; subsample=0.813; colsample_bytree=0.718; eta=0.0476; gamma=0.432; max_delta_step=6 : acc.test.mean=0.8795953
# balanced logloss: [Tune] Result: max_depth=8; subsample=0.734; colsample_bytree=0.949; eta=0.195; gamma=1.98; max_delta_step=7 : acc.test.mean=0.8541179

# fit using tuned parameters; imbalanced data
xgb_train_q13 <- xgb.DMatrix(data = as.matrix(q13[, -c(1, 34)]), label = (as.numeric(q13$q13)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0367,
  max_depth = 4,
  gamma = 2.38,
  max_delta_step = 3,
  subsample = 0.626,
  colsample_bytree = 0.887,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0366,
  max_depth = 4,
  gamma = 0.806,
  max_delta_step = 7,
  subsample = 0.704,
  colsample_bytree = 0.854,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0476,
  max_depth = 15,
  gamma = 0.432,
  max_delta_step = 6,
  subsample = 0.813,
  colsample_bytree = 0.718,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.195,
  max_depth = 8,
  gamma = 1.98,
  max_delta_step = 7,
  subsample = 0.734,
  colsample_bytree = 0.949,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q13_gb <- xgb.train(params = xgb_params, data = xgb_train_q13, nrounds = 5000, verbose = 1, maximize = FALSE)
q13_out_gb <- predict(q13_gb, as.matrix(q13_test[, -c(1, 34)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q13_out_gb > thres) + 0
  accs[i] = mean(pred == q13_test$correct)
}
plot(thresholds, accs)
q13_pred <- (q13_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q13_pred), as.factor(q13_test$correct)) 
CM
auc <- auc(q13_pred, q13_test$correct) 
auc

# record eval metrics for question 8
eval <- rbind(eval, cbind(question = 13,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 2 to results df
results[(12*n+1):(13*n), 1] <- q13_test$session_id
results[(12*n+1):(13*n), 2] <- q13_pred

### -----------------------------------------------------------
### Checkpoint 3

cp3_val <- cp3_val %>% 
  select(-c(q1, q2, q3, cp1, q4, q5, q6, q7, q8, q9, q10, 
            q11, q12, q13, cp2, q14, q15, q16, q17, q18, correctness))

### -----------------------------------------------------------
### question 14
# prepare tesing data
q14_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q14_test <- left_join(q14_test, correctness, by = "session_id")
q14_test <- left_join(q14_test, labels[(labels$question == 14) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q14[, -1], target = "q14")
testtask <- makeClassifTask(data = q14_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=8; subsample=0.995; colsample_bytree=0.99; eta=0.197; gamma=9.28; max_delta_step=8 : acc.test.mean=0.7353289
# unbalanced; logloss: [Tune] Result: max_depth=6; subsample=0.81; colsample_bytree=0.747; eta=0.0594; gamma=8.86; max_delta_step=2 : acc.test.mean=0.7395058
# balanced: [Tune] Result: max_depth=8; subsample=0.784; colsample_bytree=0.785; eta=0.241; gamma=0.29; max_delta_step=4 : acc.test.mean=0.8622438
# balanced logloss: [Tune] Result: max_depth=10; subsample=0.663; colsample_bytree=0.746; eta=0.137; gamma=1.11; max_delta_step=7 : acc.test.mean=0.8628112

# fit using tuned parameters; imbalanced data
xgb_train_q14 <- xgb.DMatrix(data = as.matrix(q14[, -c(1, 39)]), label = (as.numeric(q14$q14)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.197,
  max_depth = 8,
  gamma = 9.28,
  max_delta_step = 8,
  subsample = 0.995,
  colsample_bytree = 0.99,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0594,
  max_depth = 6,
  gamma = 8.86,
  max_delta_step = 2,
  subsample = 0.81,
  colsample_bytree = 0.747,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.241,
  max_depth = 8,
  gamma = 0.29,
  max_delta_step = 4,
  subsample = 0.784,
  colsample_bytree = 0.785,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.137,
  max_depth = 8,
  gamma = 1.11,
  max_delta_step = 7,
  subsample = 0.663,
  colsample_bytree = 0.746,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# train xgboost for question 5
set.seed(579)
q14_gb <- xgb.train(params = xgb_params, data = xgb_train_q14, nrounds = 5000, verbose = 1, maximize = FALSE)
q14_out_gb <- predict(q14_gb, as.matrix(q14_test[, -c(1, 39)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q14_out_gb > thres) + 0
  accs[i] = mean(pred == q14_test$correct)
}
plot(thresholds, accs)
q14_pred <- (q14_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q14_pred), as.factor(q14_test$correct)) 
CM
auc <- auc(q14_pred, q14_test$correct) 
auc

# record eval metrics for question 14
eval <- rbind(eval, cbind(question = 14,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 2 to results df
results[(13*n+1):(14*n), 1] <- q14_test$session_id
results[(13*n+1):(14*n), 2] <- q14_pred

### -----------------------------------------------------------
### question 15
# prepare tesing data
q15_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q15_test <- left_join(q15_test, correctness, by = "session_id")
q15_test <- left_join(q15_test, labels[(labels$question == 15) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q15[, -1], target = "q15")
testtask <- makeClassifTask(data = q15_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "logloss", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=5; subsample=0.922; colsample_bytree=0.799; eta=0.0744; gamma=7.12; max_delta_step=8 : acc.test.mean=0.6572841
# unbalanced; logloss: [Tune] Result: max_depth=7; subsample=0.973; colsample_bytree=0.88; eta=0.167; gamma=4.54; max_delta_step=9 : acc.test.mean=0.6355901

# fit using tuned parameters; imbalanced data
xgb_train_q15 <- xgb.DMatrix(data = as.matrix(q15[, -c(1, 39)]), label = (as.numeric(q15$q15)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0744,
  max_depth = 5,
  gamma = 7.12,
  max_delta_step = 8,
  subsample = 0.922,
  colsample_bytree = 0.799,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.167,
  max_depth = 7,
  gamma = 4.54,
  max_delta_step = 9,
  subsample = 0.973,
  colsample_bytree = 0.88,
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# train xgboost for question 15
set.seed(579)
q15_gb <- xgb.train(params = xgb_params, data = xgb_train_q15, nrounds = 5000, verbose = 1, maximize = FALSE)
q15_out_gb <- predict(q15_gb, as.matrix(q15_test[, -c(1, 39)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q15_out_gb > thres) + 0
  accs[i] = mean(pred == q15_test$correct)
}
plot(thresholds, accs)
q15_pred <- (q15_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q15_pred), as.factor(q15_test$correct)) 
CM
auc <- auc(q15_pred, q15_test$correct) 
auc

# record eval metrics for question 14
eval <- rbind(eval, cbind(question = 15,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "unbalanced",
                          eval_metric = "logloss"))

# add predictions of question 2 to results df
results[(14*n+1):(15*n), 1] <- q15_test$session_id
results[(14*n+1):(15*n), 2] <- q15_pred

### -----------------------------------------------------------
### question 16
# prepare tesing data
q16_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q16_test <- left_join(q16_test, correctness, by = "session_id")
q16_test <- left_join(q16_test, labels[(labels$question == 16) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q16[, -1], target = "q16")
testtask <- makeClassifTask(data = q16_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=6; subsample=0.513; colsample_bytree=0.907; eta=0.015; gamma=6.97; max_delta_step=5 : acc.test.mean=0.7336481
# unbalanced; logloss: [Tune] Result: max_depth=9; subsample=0.871; colsample_bytree=0.574; eta=0.0318; gamma=6.96; max_delta_step=7 : acc.test.mean=0.7322921
# balanced: [Tune] Result: max_depth=9; subsample=0.585; colsample_bytree=0.6; eta=0.0303; gamma=0.732; max_delta_step=6 : acc.test.mean=0.8828270
# balanced logloss: [Tune] Result: max_depth=15; subsample=0.825; colsample_bytree=0.592; eta=0.143; gamma=0.385; max_delta_step=9 : acc.test.mean=0.8937237

# fit using tuned parameters; imbalanced data
xgb_train_q16 <- xgb.DMatrix(data = as.matrix(q16[, -c(1, 39)]), label = (as.numeric(q16$q16)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.015,
  max_depth = 6,
  gamma = 6.97,
  max_delta_step = 5,
  subsample = 0.513,
  colsample_bytree = 0.907,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0318,
  max_depth = 9,
  gamma = 6.96,
  max_delta_step = 7,
  subsample = 0.871,
  colsample_bytree = 0.574,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0303,
  max_depth = 9,
  gamma = 0.732,
  max_delta_step = 6,
  subsample = 0.585,
  colsample_bytree = 0.6,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.143,
  max_depth = 15,
  gamma = 0.385,
  max_delta_step = 9,
  subsample = 0.825,
  colsample_bytree = 0.592,
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# train xgboost for question 15
set.seed(579)
q16_gb <- xgb.train(params = xgb_params, data = xgb_train_q16, nrounds = 5000, verbose = 1, maximize = FALSE)
q16_out_gb <- predict(q16_gb, as.matrix(q16_test[, -c(1, 39)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q16_out_gb > thres) + 0
  accs[i] = mean(pred == q16_test$correct)
}
plot(thresholds, accs)
q16_pred <- (q16_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q16_pred), as.factor(q16_test$correct)) 
CM
auc <- auc(q16_pred, q16_test$correct) 
auc

# record eval metrics for question 14
eval <- rbind(eval, cbind(question = 16,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 16 to results df
results[(15*n+1):(16*n), 1] <- q16_test$session_id
results[(15*n+1):(16*n), 2] <- q16_pred

### -----------------------------------------------------------
### question 17
# prepare tesing data
q17_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q17_test <- left_join(q17_test, correctness, by = "session_id")
q17_test <- left_join(q17_test, labels[(labels$question == 17) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q17[, -1], target = "q17")
testtask <- makeClassifTask(data = q17_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=3; subsample=0.524; colsample_bytree=0.567; eta=0.186; gamma=7.9; max_delta_step=4 : acc.test.mean=0.6753987
# unbalanced; logloss: [Tune] Result: max_depth=4; subsample=0.9; colsample_bytree=0.9; eta=0.0739; gamma=3.24; max_delta_step=7 : acc.test.mean=0.6755071
# balanced: [Tune] Result: max_depth=13; subsample=0.981; colsample_bytree=0.976; eta=0.0109; gamma=0.394; max_delta_step=6 : acc.test.mean=0.8345001
# balanced logloss: [Tune] Result: max_depth=12; subsample=0.553; colsample_bytree=0.748; eta=0.0571; gamma=0.656; max_delta_step=9 : acc.test.mean=0.8251830

# fit using tuned parameters; imbalanced data
xgb_train_q17 <- xgb.DMatrix(data = as.matrix(q17[, -c(1, 39)]), label = (as.numeric(q17$q17)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.186,
  max_depth = 3,
  gamma = 7.9,
  max_delta_step = 4,
  subsample = 0.524,
  colsample_bytree = 0.567,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0739,
  max_depth = 4,
  gamma = 3.24,
  max_delta_step = 7,
  subsample = 0.9,
  colsample_bytree = 0.9,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0109,
  max_depth = 13,
  gamma = 0.394,
  max_delta_step = 6,
  subsample = 0.981,
  colsample_bytree = 0.976,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0571,
  max_depth = 12,
  gamma = 0.656,
  max_delta_step = 9,
  subsample = 0.553,
  colsample_bytree = 0.748,
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# train xgboost for question 17
set.seed(579)
q17_gb <- xgb.train(params = xgb_params, data = xgb_train_q17, nrounds = 5000, verbose = 1, maximize = FALSE)
q17_out_gb <- predict(q17_gb, as.matrix(q17_test[, -c(1, 39)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q17_out_gb > thres) + 0
  accs[i] = mean(pred == q17_test$correct)
}
plot(thresholds, accs)
q17_pred <- (q17_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q17_pred), as.factor(q17_test$correct)) 
CM
auc <- auc(q17_pred, q17_test$correct) 
auc

# record eval metrics for question 17
eval <- rbind(eval, cbind(question = 17,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = as.numeric(auc),
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 17 to results df
results[(16*n+1):(17*n), 1] <- q17_test$session_id
results[(16*n+1):(17*n), 2] <- q17_pred

### -----------------------------------------------------------
### question 18
# prepare tesing data
q18_test <- cp3_val
correctness <- results %>% 
  drop_na() %>% 
  group_by(session_id) %>% 
  summarise(correctness = mean(correct)) 
q18_test <- left_join(q18_test, correctness, by = "session_id")
q18_test <- left_join(q18_test, labels[(labels$question == 18) & (labels$session_id %in% val_ids), c(1, 2)], by = "session_id")

# optimize parameters
# create tasks
traintask <- makeClassifTask(data = q18_[, -1], target = "q18")
testtask <- makeClassifTask(data = q18_test[, -1], target = "correct")
# do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)
# create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="binary:logistic", eval_metric = "auc", nrounds=1000L, booster = "gbtree")
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
parallelStartSocket(cpus = detectCores())
#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params_try, control = ctrl, show.info = T)
# unbalanced: [Tune] Result: max_depth=4; subsample=0.952; colsample_bytree=0.676; eta=0.0365; gamma=5.95; max_delta_step=1 : acc.test.mean=0.9510793
# unbalanced; logloss: [Tune] Result: max_depth=3; subsample=0.922; colsample_bytree=0.71; eta=0.267; gamma=9.43; max_delta_step=5 : acc.test.mean=0.9508082
# balanced: [Tune] Result: max_depth=10; subsample=0.774; colsample_bytree=0.537; eta=0.2; gamma=0.442; max_delta_step=2 : acc.test.mean=0.9943086
# balanced logloss: [Tune] Result: max_depth=11; subsample=0.733; colsample_bytree=0.661; eta=0.102; gamma=3.3; max_delta_step=1 : acc.test.mean=0.9901561

# fit using tuned parameters; imbalanced data
xgb_train_q18 <- xgb.DMatrix(data = as.matrix(q18[, -c(1, 39)]), label = (as.numeric(q18$q18)) - 1)
# unbalanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.0365,
  max_depth = 4,
  gamma = 5.95,
  max_delta_step = 1,
  subsample = 0.952,
  colsample_bytree = 0.676,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# unbalanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.267,
  max_depth = 3,
  gamma = 9.43,
  max_delta_step = 5,
  subsample = 0.922,
  colsample_bytree = 0.71,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
# balanced
xgb_params <- list(
  booster = "gbtree",
  eta = 0.2,
  max_depth = 10,
  gamma = 0.442,
  max_delta_step = 2,
  subsample = 0.774,
  colsample_bytree = 0.537,
  objective = "binary:logistic",
  eval_metric = "auc"
)
# balanced; logloss
xgb_params <- list(
  booster = "gbtree",
  eta = 0.102,
  max_depth = 11,
  gamma = 3.3,
  max_delta_step = 1,
  subsample = 0.733,
  colsample_bytree = 0.661,
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# train xgboost for question 17
set.seed(579)
q18_gb <- xgb.train(params = xgb_params, data = xgb_train_q18, nrounds = 5000, verbose = 1, maximize = FALSE)
q18_out_gb <- predict(q18_gb, as.matrix(q18_test[, -c(1, 39)]))
# find optimal threshold
thresholds <- seq(0, 1, by = 0.01)
accs <- numeric(length(thresholds))
for (i in 1:length(thresholds)) {
  thres = thresholds[i]
  pred = (q18_out_gb > thres) + 0
  accs[i] = mean(pred == q18_test$correct)
}
plot(thresholds, accs)
q18_pred <- (q18_out_gb > thresholds[which.max(accs)]) + 0
CM <-confusionMatrix(as.factor(q18_pred), as.factor(q18_test$correct)) 
CM
auc <- auc(q18_pred, q18_test$correct) 
auc

# record eval metrics for question 17
eval <- rbind(eval, cbind(question = 18,
                          accuracy = CM$overall[1] %>% as.vector(),
                          auc = NA,
                          data = "balanced",
                          eval_metric = "auc"))

# add predictions of question 17 to results df
results[(17*n+1):(18*n), 1] <- q18_test$session_id
results[(17*n+1):(18*n), 2] <- q18_pred








