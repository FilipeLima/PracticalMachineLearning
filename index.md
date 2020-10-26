Background
----------

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is
now possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here: [HAR
Project](http://groupware.les.inf.puc-rio.br/har) (see the section on
the Weight Lifting Exercise Dataset).

Data cleaning
-------------

First we will find the most relevant variables in the [Training
data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
for the project. We chose the ones that have the results for roll, yaw,
pitch, gyros, accel and magnet for belt, forearm, arm and dumbell, and
of course, the classe variable. We based our choice using the
description of the variables in this
[article](http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf)
by Velloso *et al*. Then we will create a new dataset using those
variables and classe as a factor.

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(parallel) 
    library(doParallel) #For parallel computing

    ## Loading required package: foreach

    ## Loading required package: iterators

    WLEdataframe <- read.csv("pml-training.csv")
    WLEdataframe$classe <- as.factor(WLEdataframe$classe)
    #The index of the variables we want
    Selection1 <- c(8:10,37:48,60:68,84:86,113:124,151:160)
    WLEdataframe <- WLEdataframe[,Selection1]

The variables in this new dataframe are:

    names(WLEdataframe)

    ##  [1] "roll_belt"         "pitch_belt"        "yaw_belt"         
    ##  [4] "gyros_belt_x"      "gyros_belt_y"      "gyros_belt_z"     
    ##  [7] "accel_belt_x"      "accel_belt_y"      "accel_belt_z"     
    ## [10] "magnet_belt_x"     "magnet_belt_y"     "magnet_belt_z"    
    ## [13] "roll_arm"          "pitch_arm"         "yaw_arm"          
    ## [16] "gyros_arm_x"       "gyros_arm_y"       "gyros_arm_z"      
    ## [19] "accel_arm_x"       "accel_arm_y"       "accel_arm_z"      
    ## [22] "magnet_arm_x"      "magnet_arm_y"      "magnet_arm_z"     
    ## [25] "roll_dumbbell"     "pitch_dumbbell"    "yaw_dumbbell"     
    ## [28] "gyros_dumbbell_x"  "gyros_dumbbell_y"  "gyros_dumbbell_z" 
    ## [31] "accel_dumbbell_x"  "accel_dumbbell_y"  "accel_dumbbell_z" 
    ## [34] "magnet_dumbbell_x" "magnet_dumbbell_y" "magnet_dumbbell_z"
    ## [37] "roll_forearm"      "pitch_forearm"     "yaw_forearm"      
    ## [40] "gyros_forearm_x"   "gyros_forearm_y"   "gyros_forearm_z"  
    ## [43] "accel_forearm_x"   "accel_forearm_y"   "accel_forearm_z"  
    ## [46] "magnet_forearm_x"  "magnet_forearm_y"  "magnet_forearm_z" 
    ## [49] "classe"

With this line of code we can confirm that there are no NA values in
this new dataset.

    sum(is.na(WLEdataframe))

    ## [1] 0

Splitting and training
----------------------

We split the dataset using 75% for training. We use cross validation as
a method of train control and a 10-fold cross validation. Then, we
trained the data using the Random Forest algorithm. But first, we are
going to create a cluster to optimize our computing time.

    cluster <- makeCluster(detectCores() - 1) #Getting all the cores but one
    registerDoParallel(cluster) #Registering clusters

Now, training the model with the above parameters, we have:

    set.seed(20201025)
    inTraining <- createDataPartition(y=WLEdataframe$classe, p = .75, list=FALSE)
    training <- WLEdataframe[inTraining,]
    testing <- WLEdataframe[-inTraining,]
    fitControl <- trainControl(method = "cv",number = 10)
    modRF <- train(training[,-49],training[,49],data=training,method="rf",
                   trControl = fitControl)
    modRF

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    48 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 13245, 13247, 13247, 13246, 13246, 13245, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9934096  0.9916626
    ##   25    0.9936817  0.9920072
    ##   48    0.9875667  0.9842707
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 25.

The next chunk of code is used to close the cluster.

    stopCluster(cluster)
    registerDoSEQ()

Testing and Results
-------------------

Now, we will test the model in the testing data and look at the
confusion matrix.

    predictions <- predict(modRF, newdata=testing)
    confusionMatrix(predictions, testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    5    0    0    0
    ##          B    0  943    2    0    0
    ##          C    1    1  851    4    0
    ##          D    0    0    2  800    3
    ##          E    0    0    0    0  898
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9963          
    ##                  95% CI : (0.9942, 0.9978)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9954          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9937   0.9953   0.9950   0.9967
    ## Specificity            0.9986   0.9995   0.9985   0.9988   1.0000
    ## Pos Pred Value         0.9964   0.9979   0.9930   0.9938   1.0000
    ## Neg Pred Value         0.9997   0.9985   0.9990   0.9990   0.9993
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2843   0.1923   0.1735   0.1631   0.1831
    ## Detection Prevalence   0.2853   0.1927   0.1748   0.1642   0.1831
    ## Balanced Accuracy      0.9989   0.9966   0.9969   0.9969   0.9983

As the accuracy is 99.63%, the Out-of-the-sample error is 1 minus the
accuracy, which is 0.37%.

The Course Project Prediction Quiz
----------------------------------

Now, we will use our model to predict the classes for this [Testing
Dataset](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv),
with 20 observations.

    predictDATAFRAME <- read.csv("pml-testing.csv")
    predict(modRF, newdata=predictDATAFRAME)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Checking this result in the quiz, we got 100% of the problems correct.
