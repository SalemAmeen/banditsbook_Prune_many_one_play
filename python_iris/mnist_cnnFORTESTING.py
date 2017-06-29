# FOR MAC PC
#execfile("/Users/salemameen/Desktop/banditsbook/python/core.py")
exec(open("modelUCB1.py").read())
import matplotlib.pyplot as plt
import time

# FOR UBUNTU PC
#execfile("/home/sal/Desktop/banditsbook/python/core.py")
#execfile("/home/sal/Desktop/banditsbook/python/modelUCB1.py")


#algo = UCB1([], [])
#alpha=0.3 # alpha in [0.1, 0.3, 0.5, 0.7, 0.9]
#algo = UCB2(alpha, [], [])
#algo = AnnealingEpsilonGreedy([], [])
#epsilon = 0.9
#algo = EpsilonGreedy(epsilon, [], [])

#exp3_gamma = 0.2 #exp3_gamma in [0.1, 0.2, 0.3, 0.4, 0.5]
#algo = Exp3(exp3_gamma, [])
#temperature = 0.9
#algo = Softmax(temperature, [], [])
#algo = AnnealingSoftmax([], [])
#eta = 0.9  # eta in [.5, .8, .9, 1, 2]
#algo = Hedge(eta, [], []) # not working right now
#print algo
#arms=[1,2,3,4,5]
num_sims=1 # How many times want to play at one time. With Particulr arm a
           # How many other arms or weights want to check to see if they work
           # well with this arm or weigh.
###############################################
horizon=50 # Playing times

MaxofPrune = 16
######################################
#arms=FC_weights_3
arms=np.arange(col)
#arms=np.ravel(arms)
#rewards = [0.0 for i in range(row,col)]
#rewards= [[0 for x in range(col)] for y in range(row)]
#rewards=np.zeros((row,col))
rewards=np.zeros(col)
AccuracyAftrerPrune=np.zeros((MaxofPrune))
cumulative_rewards = [0.0 for i in range(horizon)]
algo.initialize(len(arms))
start_time = time.time()

for t in range(horizon):

    model.set_weights(All_weights_BUCKUP)
    All_weights=model.get_weights()
    FC_weights_3=All_weights[0]

    chosen_arm = algo.select_arm()
    #print 'chosen_arm', chosen_arm
    index=chosen_arm
    #index=np.unravel_index(chosen_arm,FC_weights_3.shape)
    #print 'index', index
    #print 'horizon = ', t
    #i=index[0]
    #j=index[1]
    #temp=FC_weights_3[:,index]
    FC_weights_3[:,index]=0
    All_weights[0]=FC_weights_3
    model.set_weights(All_weights)
    ###################################################
    score = model.evaluate(X_test, labelsVal, verbose=0)
    ###################################################
    #score = model.evaluate(X_deploy, labelsTest, verbose=0)
    ########################################################
    NewAccuracy = score[1]
    np.save('AccuracyBeforePruning' , NewAccuracy-0.01)
    delta = NewAccuracy - OldAccuracy
    reward=max(0,delta+0.2)
    #if NewAccuracy >= OldAccuracy-0.01:
    #if NewAccuracy >= OldAccuracy:
        #reward = 1
    #else:
        #reward = 0
    #print 'reward =',reward
    rewards[index] =rewards[index] + reward
    if t == 1:
        cumulative_rewards[t] = reward
    else:
        cumulative_rewards[t] = cumulative_rewards[t - 1] + reward
    algo.update(chosen_arm, reward)
    #FC_weights_3[:,index]=temp
# Start pruning
print("The time for running this method is %s seconds " % (time.time() - start_time))
np.save('rewards',rewards)

model.set_weights(All_weights_BUCKUP)
All_weights=model.get_weights()
FC_weights_3=All_weights[0]


print('Finsh playing start pruining:')
for t in range(MaxofPrune):

    x=np.argmax(rewards)
    #print 'REWARD = ', rewards[x]
    #if rewards[x]==0:
        #break
    rewards[x]=-100
    #idx=np.unravel_index(x,rewards.shape)
    idx=x
    #print 'x = ', idx

    #i=index[0]
    #j=index[1]
    FC_weights_3[:,idx]=0
    All_weights[0]=FC_weights_3
    model.set_weights(All_weights)
    #print 'Number of pruning = ', t
    score = model.evaluate(X_deploy, labelsTest, verbose=0)
    AccuracyAftrerPrune[t] = score[1]
    print('Test accuracy after pruning:', score[1])
     #if  score[1] > (OldAccuracy - 0.01):
		#print 'save'
    spam = path + 'spam'+ str(t)
    accurMat = path + 'Accuracy'+ str(t)
    model.save_weights(spam+'.hdf5', overwrite=True)
    np.save(accurMat,AccuracyAftrerPrune[t])
    ############

plt.plot(AccuracyAftrerPrune)
plt.xlabel('Number of Neurons  pruned')
plt.ylabel('Accuracy after pruning')
plt.title(Alg_name)
plt.axis([0, 16, 0, 1])
plt.grid(True)
plt.show()

AccuracyAftrerPruneMat = path + 'AccuracyAftrerPrune'
np.save(AccuracyAftrerPruneMat , AccuracyAftrerPrune)
