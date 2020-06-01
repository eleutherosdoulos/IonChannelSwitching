###What I Know###
#f1 scores are equally weighted per open channel number
#signal vs time seems to follow horizontal line or parabola generally
#open channels are generally banded by signal
#signal vs open channel seems to follow the same distribution within an experiment
#mean signal value of a channel isn't very consistent across experiments
#signal range per channel same within an experiment
#channels open and close rapidly and repeatedly
#signal is pA

#In terms of relative performance, note that whilst there was a strong correlation
#between open probabilities measured between Deep-Channel and threshold crossing, 
#the F1-accuracy scores of the noisy data measured by threshold crossing fell off sharply. 
#The significance of this is that whilst average open probability estimates from threshold 
#crossing seem reasonable (some over estimates, some under-estimates cancelling out) the 
#time-point by time-point accuracy essential for kinetic analyses is poor.
#https://www.nature.com/articles/s42003-019-0729-3#Tab2

###Questions I Have###
#Should I bootstrap my train set or set aside 10 percent for cross validation?
#What's the best way to deal with various experiments?
  #(experiment ensemble, tag experiment categorically, tag experiment with mean/sd)
#What's the max number of open channels possible?
#SKM using QuB13 and Minimum description length (MDL, using MatLab)?


###Things to Do###
#refactor cleaning function
#Fix legend icon size on summary graph

###Possible modes of attack
#simple regression prediction subtracted to flatten, signal range to estimate channel number, divide
#mash a neural net on it
#FT for signals

library(ggplot2)
library(caret)
library(tidyverse)
library(RColorBrewer)
library(tensorflow)
library(keras)

install_tensorflow()
install_keras()
#Functions------------------------------

#loss function
macro_f1<-function(predicted,actual){
  eval_stats<-confusionMatrix(as.factor(predicted), as.factor(actual), mode = "prec_recall", positive="1")
  mean(eval_stats$byClass[,7])
}

#data prep function
clean_data<-function(data){
  new_columns<-data%>%mutate(
    batch=floor((train_data$time-.0001)/50)+1, 
    adjustedtime=train_data$time-((batch-1)*50))
  
  averages<-new_columns%>%
    group_by(batch)%>%
    summarize(signal_mean=mean(signal))
  
  new_columns%>%left_join(averages,by= "batch")
}


train_data<-read.csv("train.csv")
test_data<-read.csv("test.csv")

clean_train_data<-clean_data(train_data)
clean_test_data<-clean_data(test_data)

#Exploratory Analysis--------------------------------

#signal summary stats
signal_summary<-clean_train_data%>%
  group_by(batch)%>%
  summarize("Signal Mean"=mean(signal), 
            "Signal Median"=median(signal), 
            "Signal Standard Deviation"=sd(signal))

signal_spread<-clean_train_data%>%
  group_by(batch)%>%
  summarize(sd=sd(signal), 
            max=max(open_channels), 
            range=max(open_channels)-
            min(open_channels),
            sd_channels=sd(open_channels))

#channel density
open_channel_summary<-clean_train_data%>%
  group_by(batch,open_channels)%>%
  summarize(proportion=n()/500000)

spread_open_channel_summary<-open_channel_summary%>%
  spread(open_channels,proportion,0)

open_channel_heatmap<-open_channel_summary%>%
  ggplot(aes(x=open_channels, y=batch, fill= proportion))+ 
  geom_tile()+
  scale_fill_distiller(palette="Spectral")+
  scale_y_continuous(name="Experiment", breaks= c(10:1),labels=c(10:1))+
  scale_x_continuous(name="Number of Open Channels", breaks= c(0:10),labels=c(0:10))+
  ggtitle("Proportion of Open Channels per Patch Clamp Experiment")

#signal density by channel
signal_channel_means<-clean_train_data%>%
  group_by(batch,open_channels)%>%
  summarize(mean=mean(signal))

spread_signal_channel_means<-signal_channel_means%>%
  spread(open_channels,mean,".")

signal_channel_means_heatmap<-signal_channel_means%>%
  ggplot(aes(x=open_channels, y=batch, fill=mean))+ 
  geom_tile()+
  scale_fill_distiller(palette="Spectral")+
  scale_y_continuous(name="Experiment", breaks= c(10:1),labels=c(10:1))+
  scale_x_continuous(name="Number of Open Channels", breaks= c(0:10),labels=c(0:10))+
  ggtitle("Mean Signal Value of Open Channel Number per Experiment")

#signal sd by channel
signal_channel_sds<-clean_train_data%>%
  group_by(batch,open_channels)%>%
  summarize(sd=sd(signal))

spread_signal_channel_sds<-signal_channel_sds%>%
  spread(open_channels,sd,".")

signal_channel_sds_heatmap<-signal_channel_sds%>%
  ggplot(aes(x=open_channels, y=batch, fill=sd))+ 
  geom_tile()+
  scale_fill_distiller(palette="Spectral")+
  scale_y_continuous(name="Experiment", breaks= c(10:1),labels=c(10:1))+
  scale_x_continuous(name="Number of Open Channels", breaks= c(0:10),labels=c(0:10))+
  ggtitle("Standard Deviation of Open Channel Signal Value per Experiment")

#signal by channel data
signal_channel_summary<-clean_train_data%>%
  group_by(batch,open_channels)%>%
  summarize("Signal Mean"=mean(signal), 
            "Signal Median"=median(signal), 
            "Signal Standard Deviation"=sd(signal))%>%
  arrange(open_channels)

#Signal Sd vs Open Channel Range per Batch (perhaps refactor with signal range minus outliers)
sdchandata<-clean_train_data%>%
  group_by(batch)%>%
  summarize(sd=sd(signal),
            channel_range=max(open_channels)-min(open_channels))

spread_graph<-ggplot(sdchandata,aes(x=channel_range,y=sd,label=as.factor(batch)))+
  geom_point()+
  geom_text(position="jitter")
  
#signal vs time with channel color
summary_graphs<-ggplot(clean_train_data, aes(x=adjustedtime,y=signal, color=as.factor(open_channels)))+
    geom_point(shape=".")+
    ylab("Signal (current)")+
    xlab("Time (s)")+
    ggtitle("Patch Clamp Signal vs Time per Experiment")+
    scale_color_discrete(name="Open Channels", labels, 
      limits=c("0","1","2","3","4","5","6","7","8","9","10"))+
    guides(shape = guide_legend(override.aes = list(size=10)))+
    facet_wrap(batch~.)

#signal vs open channels violin plots
sig_v_open_channels_graphs<-ggplot(clean_train_data, aes(x=as.factor(open_channels),y=signal))+
  geom_violin(scale="count")+
  ylab("Signal (current)")+
  xlab("Number of Open Channels")+
  ggtitle("Patch Clamp Signal vs Number of Open Channels per Experiment")+
  facet_wrap(batch~.)

#equal area signal vs open channels violin plots
sig_v_open_channels_graphs_equal_area<-ggplot(clean_train_data, aes(x=as.factor(open_channels),y=signal))+
  geom_violin()+
  ylab("Signal (current)")+
  xlab("Number of Open Channels")+
  ggtitle("Patch Clamp Signal vs Number of Open Channels per Experiment, Equal Area")+
  facet_wrap(batch~.)

#time vs open channels violin plots
open_channels_v_t_graphs<-ggplot(clean_train_data, aes(x=adjustedtime,y=as.factor(open_channels)))+
  geom_violin(scale="count")+
  xlab("Time (s)")+
  ylab("Number of Open Channels")+
  ggtitle("Number of Open Channels vs Time per Experiment")+
  facet_wrap(batch~.)

#time vs open channels violin plots, equal area
open_channels_v_t_graphs_equal_area<-ggplot(clean_train_data, aes(x=adjustedtime,y=as.factor(open_channels)))+
  geom_violin()+
  xlab("Time (s)")+
  ylab("Number of Open Channels")+
  ggtitle("Number of Open Channels vs Time per Experiment")+
  facet_wrap(batch~.)

#Fit Functions---------------------------------
#straight up random guessing
random_guess<-sample(0:10, 5000000, replace=TRUE)
random_f1<-macro_f1(random_guess,train_data$open_channels)
random_f1 #.0779


#try stepwise linear and quadratic models (this could be more generalized)
#subtract regression prediction from model to flatten
#use range signal vs range open channels to find number of channels (clean up with NoiseFiltersR?)
#divide starting from zero at the bottom, channels are equal sig range
#figure out if zero should be lowest or no

#I had to hardcode the predict line, I couldn't find the logic flaw in start and stop time
piecewise<-function(signal_data,chunk_seconds){
  
  chunks<-round(max(signal_data$adjustedtime)/chunk_seconds)
  interval<-signal_data$adjustedtime[2]-signal_data$adjustedtime[1]
  
  fits<-sapply(1:chunks,function(c){
    start_t<-((c-1)*chunk_seconds)+interval
    end_t<-(c*chunk_seconds)
    data<-filter(signal_data,between(adjustedtime,start_t,end_t))
    
    fit<-lm(signal~adjustedtime, data=data)
    predict(fit, newdata=data.frame(adjustedtime=seq(((c-1)*10+.0001),(c*10),.0001)))
  })
  
  unlist(fits)
}

flattener<- function(signal_data){
  #a vector of batch numbers for the experiments
  batch_range<-min(signal_data$batch):max(signal_data$batch)
  
  #a linear step function fitting the 50s experiments in 5 chunks
  piecewise_functions<-sapply(batch_range,function(y){
    piecewise(filter(signal_data,batch==y),10)
  })
    
  #the RMSE of each piecewise function fit
  piecewise_functions_errs<-sapply(batch_range,function(z){
    data<-filter(signal_data,batch==z)
    RMSE(piecewise_functions[[z]],data$signal)
    })
  
  #a quadratic function for each experiment
  quadratic_functions<-sapply(batch_range,function(y){
    fit<-lm(signal~poly(adjustedtime,2), data=filter(signal_data,batch==y))
    as.numeric(unlist(fit[5]))
    })
  
  #the RMSE for each quadratic function fit
  quadratic_functions_errs<-sapply(batch_range,function(z){
    data<-filter(signal_data,batch==z)
    RMSE(quadratic_functions[,z],data$signal)
    })
  
  #selecting which of the two fits was the best for each experiment
  predictions<-sapply(batch_range,function(z){
    if(piecewise_functions_errs[z]<quadratic_functions_errs[z]){
      piecewise_functions[[z]]
    }
    else{
      quadratic_functions[,z]
    }
  })
  
  #putting the predictions together as a single vector
  fits<-as.vector(as.matrix(predictions[c(1:length(predictions))]))
  
  #subtracting the predictions from the actual signal to flatten the graphs
  flatvalues<-signal_data$signal-fits
  
  #adding these flattened predictions
  mutate(signal_data,flatvalues=flatvalues)
}

#using the above function on train data
flattened_signal<-flattener(clean_train_data)

#filtering out extreme outlier signal data
filtered_flat<-flattened_signal%>%
  group_by(batch)%>%
  filter(abs(flatvalues)<(mean(flatvalues)+(7*sd(flatvalues))))

#this will eventually be turned in to a linear equation that can be used with the flatten and filter output of test data
sdchandata2<-filtered_flat%>%
  group_by(batch)%>%
  summarize(sig_range=max(flatvalues)-min(flatvalues),
            channel_range=max(open_channels)-min(open_channels))

spread_graph2<-ggplot(sdchandata2,aes(x=sig_range,y=channel_range,label=as.factor(batch)))+
  geom_point()+
  geom_text(position="jitter")

spread_graph2

#neural net, man
x_train<-select(clean_train_data,c(adjustedtime,signal,signal_mean))
x_train$adjustedtime=x_train$adjustedtime/50
x_train$signal=(x_train$signal-min(x_train$signal))/(max(x_train$signal-min(x_train$signal)))
x_train$signal_mean=(x_train$signal_mean-min(x_train$signal_mean))/(max(x_train$signal_mean-min(x_train$signal_mean)))
x_train=data.matrix(x_train)

y_train<-as.factor(clean_train_data$open_channels)
y_train<-to_categorical(y_train,11)

model <- keras_model_sequential()%>% 
  # Adds a densely-connected layer with 64 units to the model:
  layer_dense(units = 64, activation = 'relu') %>%
  
  # Add another:
  layer_dense(units = 64, activation = 'relu') %>%
  
  # Add a softmax layer with 10 output units:
  layer_dense(units = 10, activation = 'softmax')

#random forest, final model
controlrf <- trainControl(method = "oob",
                          number= 10,
                          p=0.75)
mtry<-data.frame(mtry=1:10)
train_rf <- train(y_train ~ x_train, method="rf",
              data = prep,
              tuneGrid=mtry,
              trControl = controlrf)
