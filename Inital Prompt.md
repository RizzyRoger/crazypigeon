# Context:
1. I want to train an LLM through reinforcement learning. The AI should output text, and it should be reinforced at random intervals at time. This is to recreate B.F. Skinner's finding in his paper "Superstision in the Pigeon"
2. This should be non-contingent reinforcement and completely random!
# Features of the app:
1. This bot should have a good understanding of vocabulary, but not too much so it can be influenced by our reinforcement. before we start reinforcement.
2. For this, we can use NanoGPT and train it on english data. 
3. After it appears understands english to an extend, we need a point that we can define that it is read to start testing
4. We need to gather data on behaviors, so that model should be saved for later
5. Then, after we have all of that, we should now be able to run an experiment with copies of the model
# The Experiment:
1. First, we need to run the saved model to get some data. This should be saved as validation data and be used to measure the loss once we are done with the random reinforcements
2. The model should output text once every second. At a random interval, between (1-30)(i) seconds, it will reward the bot
3. After this has gone through several hours of training(i), we should now allow it to output text based on all the reinforcements.
4. After that, like skinner, we should allow the model some time to run withought the random reinforcement, to see if the effects last.
5. Using the saved data from the trained model, we can calculate the loss value between the randomly reinforced model and the previous version.
6. From there, we have a lot of data we can work with, output it in a file that is easy for me to analyse.
# LLM flow:
1. Data for training(i) still TBD. Recommend me some free opensouce datasets I can use. 
2. Tell me what scripts I need to run in order to start the training.
# Scripts:
1. There should be a script to commence training the model onto its baseline
2. When we have reached our desired loss value(i) for our baseline, we now can run another script that allows us to actually go on with the random reinforcement.
3. When random reinforcement is done, we need a script to run withought reinforcement, to see if the effects last.
4. Finally, we need a script to compare the difference after the random reinforcement and before it. 

# Importaint!
1. (i) Wherever you see (i), the value I have provided is not final. Work with me to find the best possible value for all of these.