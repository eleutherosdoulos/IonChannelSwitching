###What I Know###
#f1 scores are equally weighted per open channel number
#signal vs time seems to follow horizontal line or parabola generally
#open channels are generally banded by signal
#signal vs open channel seems to follow the same distribution within an experiment
#mean signal value of a channel isn't very consistent across experiments
#sd signal value of a channel is pretty consistent within experiments
#channels open and close rapidly and repeatedly
#signal is pA
#current change should be the same per channel per experiment

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
#Read paper to learn about data set generation
#look at Kaggle notebooks for ideas
#refactor cleaning function
#Fix legend icon size on summary graph
#Get RStudio EC2 working for training

###Possible modes of attack
#fit a big, general glm for each experiment, use sd to define number of levels, shift
# multiclass logistic regression if we know the levels
#neural net?

#library(randomForest)
#library(rpart)
#library(data.table)
library(ggplot2)
library(caret)
library(tidyverse)
library(RColorBrewer)

#Functions------------------------------

#loss function
macro_f1<-function(predicted,actual){
  eval_stats<-confusionMatrix(as.factor(predicted), as.factor(actual), mode = "prec_recall", positive="1")
  mean(eval_stats$byClass[,7])
}

#data prep function
clean_data<-function(data){
  data%>%mutate(
    batch=floor((train_data$time-.0001)/50)+1, 
    adjustedtime=train_data$time-((batch-1)*50))
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

#multiclass logistic regression
#knn for smoothing?
#what kinds of neural nets can i do here?
