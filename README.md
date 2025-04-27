# cs285-homework-5-exploration-solved
**TO GET THIS SOLUTION VISIT:** [CS285 Homework 5-Exploration Solved](https://www.ankitcodinghub.com/product/cs285-cs294-112-deep-reinforcement-learning-hw5-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;111952&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS285 Homework 5-Exploration Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
1 Exploration

Exploration—how agents discover actions that lead to high rewards—is a key component of reinforcement learning. In this homework, you will investigate count-based exploration methods that modify the reward function to encourage exploring novel parts of the state space:

R˜(st) = R(st) + α · B(N(st)). (1)

N(st) represents the number of times the agent has visited the state, and the function B is a monotonically decreasing function of N(st), known as the exploration bonus. The intuition is that we would like to encourage the agent to visit novel states. If the state s is novel or is rarely visited, then N(s) will be low, and B(N(st)) will be high. Conversely, if the state s is visited often, then N(s) will be high, and B(N(st)) will be low. Therefore, the exploration bonus is an additional term to the reward function that encourages the agent to spend more time visiting novel states. The hyperparameter α indicates how much to reward novel states.

In the discrete case, we can use a histogram to keep track of the number of times the agent visited state s, so the histogram directly gives us N(st). However, when the state space is continuous, the probability of any two states being equal is 0, so we cannot simply tally the number of times we’ve visited the state. Instead, we must fit a density model fφ(st) over the state space and derive the count N(st) from fφ. Intuitively, if similar states to st have been visited many times, then fφ(st) will be high.

Given Eqn. 1, you can then run your standard reinforcement learning algorithms with only a single additional step: computing B(N(st)) as your agent acts in the environment. To do this we need to keep a replay buffer R that stores the states the agent has visited so far (note that here we only store states, not entire transitions). In the discrete case, the histogram can take place of the replay buffer; in the continuous case, the replay buffer serves as the data distribution with which we will fit the density model fφ(st). The algorithm is summarized below:

There are many possible ways to specify B(N(st)). In this homework, for discrete states we will use

.

For continuous states we will use a heuristic bonus

B(N(st)) = −logfφ(st)

which skips computing N(st) but is still a function that decreases the more

states similar to st have been visited.

1.1 Discrete States

The purpose of this section is to focus on modifying the rewards with the exploration bonus without having to worry about fitting a density model. Therefore we will modify the rewards like so:

(2)

1.2 Continuous States

Now that we have implemented the framework for Algorithm 1 for discrete states, we will now replace the histogram with a replay buffer and a density model fφ, and our goal is to be able to compute fφ(s) for any state s such that we modify the rewards like so:

R0(s,a) = R(s,a) + α(−logfφ(s)) (3)

1.2.1 Non-parametric density estimation: kernel density estimation (KDE)

Kernel density estimation is a non-parametric method that estimates the density model by maintaining a dataset of all encountered states (the replay buffer R in our case and using a kernel function Kφ(s1,s2) to measure the similarity between states.

Using an radial basis function kernel (https://en.wikipedia.org/wiki/Radial_ basis_function_kernel), we can to estimate the density of a new datapoint s by plopping a Gaussian distribution centered around each of the datapoints in R, evaluate the probability of s under each of these Gaussians, and average these probabilities together (See https://en.wikipedia.org/wiki/Kernel_ density_estimation for some nice intuitive figures). Intuitively, if a lot of the datapoints in R are close together, then the probability density of nearby points are similar because each Gaussian contributes to the probability density of these points. In particular, for a given state s, we can estimate its probability density as

.

1.2.2 Parametric density estimation: exemplar models

The problem with kernel density estimators is that to every time we evaluate the probability of a point, we have to apply the kernel to every point in the replay buffer, which becomes computationally intensive with a large replay buffer. Alternatively, we can use a parametric density estimator, which does not require a full pass through all the data to compute probabilities, but this comes at the cost of training the density model from samples, which introduces another layer of approximation.

One way to estimate the probability density fφ(s) is to train a state-conditioned noisy discriminator Ds(s0) to output 1 if s = s0 and 0 if s 6= s0 (note that Ds is a discriminator conditioned on the exemplar s, so Ds and Ds0 are not the same). The output of the discriminator is the probability that a Bernoulli random variable y takes the value 1: p(y = 1|s,s0) := p(s = s0). Then we can estimate fφ(s) by evaluating Ds on its own state s:

(4)

the reasoning behind which you can find here: https://arxiv.org/abs/1703. 01260. With this discriminator, we can estimate a probability density model over the states we’ve seen before (in the replay buffer) by training the discriminator to distinguish between exemplar states s and the states s0 from the replay buffer. Intuitively, if Ds(s0) is high, then this means that s is easily distinguished from states s0 in R, which means the probability is low that a state similar to s is in R, in which case fφ(s) is low. Conversely, if states similar to s are very common in R, then the Ds will have a hard time distinguishing s and s0, in which case Ds(s0) will output a value close to 0.5, which would make fφ(s) high.

To illustrate this, let’s consider an environment with states A, B for simplicity. Assume following two scenarios:

Scenario 1 Scenario 2

New batch of data A A

Replay Buffer B,B,B,B A,A,B,B

In Scenario 1, A is a novel state, whereas in Scenario 2 it is not. In EX2 we use examples from the new batch of data as positives and examples from the replay buffer as negatives. In Scenario 1, DA would get perfect accuracy and output 1, whereas in Scenario 2, DA would output 0.5. By plugging these values in Equation 4 one can see that in Scenario 1, fφ(A) = 0 is low, meaning that this is a new state, and in Scenario 2, fφ(A) = 1 is high, meaning that this state has been seen before.

Letting s1 := s and s2 := s0 for clarity, the discriminator can be viewed as a graphical model decomposed as:

p(y|s1,s2) = Ez1∼qz1|s1,z2∼qz2|s2 [p(y|z1,z2)q1(z1|s1)q2(z2|s2)]

where z are latent Gaussian random variables and y is a Bernoulli variable. The z’s introduce noise in the discriminator to prevent it from overfitting and encourage it to assign similar probability density to similar states. The discriminator is trained to maximize the following objective:

where

KL := β (DKL (q(z1|s1)||p(z1)) + DKL (q(z2|s2)||p(z2)))

and where p(z) is a multivariate standard Gaussian, β is a weighting coefficient that controls how much the discriminator overfits (tries to maximize the log likelihood more) or underfits (tries to make the latent distribution as close to a standard Gaussian as possible), and p(˜s) is the data distribution the discriminator is trained on, which contains half exemplar states and half replay-buffer states.

1.3 Code

1.3.1 Installation

Obtain the code from https://github.com/berkeleydeeprlcourse/homework_ fall2019/tree/master/hw5. In addition to the installation requirements from previous homeworks, install additional required packages by running: pip install

-r requirements.txt. To setup the package run python setup.py develop from the hw5 folder.

1.3.2 Overview

You will modify the following files:

• train ac exploration f18.py

• density model.py

• exploration.py

You should also familiarize yourself with the following files:

• replay.py

• pointmass.py

• sparse half cheetah.py

All other files are optional to look at.

1.4 Implementation

For problems 1 through 3, you will be working with a PointMass environment, where the agent is a dot that tries to go from location (2,2) to (18,18) of a (20,20) grid. After training has completed, you can run the following command to plot a gif of the exploration progress. python pointmass.py &lt;dirname&gt;

Problem 1

What you will implement: The reward modification (Eqn. 1), the count-based

reward bonus (Eqn. 1), and the histogram density model .

Where in the code to implement: All parts of the code where you find

### PROBLEM 1

### YOUR CODE HERE

Implementation details are in the code.

How to run: Run the commands under P1 Hist PointMass in run all.sh to compare an agent with histogram-based exploration and an agent with no exploration. Then use plot.py to plot the returns of the runs.

What will be outputted: A plot with 2 curves comparing an agent with histogram-

based exploration and an agent with no exploration.

What will a correct implementation output: The table below shows what the

reference solution gets for the mean average return when run with 8 random seeds.

Iteration Histogram No-Exploration

The table below shows what the reference solution gets for the average return one standard deviation below the mean when run with 8 random seeds.

Iteration Histogram No-Exploration

You only need to run with the three random seeds given to you in the code. Your curves should likely be comparable to the above.

Problem 2

What you will implement: The heuristic reward bonus (Eqn. 1), and the kernel

density estimator with the radial basis function kernel.

Where in the code to implement: All parts of the code where you find

### PROBLEM 2

### YOUR CODE HERE

Implementation details are in the code.

How to run: Run the commands under P2 RBF PointMass in run all.sh Then use plot.py to plot the returns of the runs to compare an agent with KDEbased exploration and an agent with no exploration (the run of which you can reuse from Problem 1)

What will be outputted: A plot with 2 curves comparing an agent with KDE-

based exploration and an agent with no exploration.

What will a correct implementation output: The table below shows what the

reference solution gets for the mean average return when run with 8 random seeds.

Iteration RBF No-Exploration

The table below shows what the reference solution gets for the average return one standard deviation below the mean when run with 8 random seeds.

Iteration RBF No-Exploration

You only need to run with the three random seeds given to you in the code. Your curves should likely be comparable to the above.

Problem 3

What you will implement: The EX2 discriminator.

Where in the code to implement: All parts of the code where you find

### PROBLEM 3

### YOUR CODE HERE

Implementation details are in the code.

How to run: Run the commands under P3 EX2 PointMass in run all.sh Then use plot.py to plot the returns of the runs to compare an agent with EX2-based exploration and an agent with no exploration (the run of which you can reuse from Problem 1)

What will be outputted: A plot with 2 curves comparing an agent with EX2-

based exploration and an agent with no exploration.

What will a correct implementation output:

The table below shows what the reference solution gets for the mean average return when run with 8 random seeds.

Iteration EX2 No-Exploration

The table below shows what the reference solution gets for the average return one standard deviation below the mean when run with 8 random seeds.

Iteration EX2 No-Exploration

You only need to run with the three random seeds given to you in the code. Your curves should likely be comparable to the above.

Problem 4

What you will implement: Nothing! Nothing at all!

How to run: Run the commands under P4 HalfCheetah in run all.sh. We have two hyperparameter settings for the EX2-based exploration. One uses the bonus coefficient α = 0.0001 and trains the density model for 10000 iterations. The other uses a bonus coefficient α = 0.001 and trains the density model for 1000 iterations. Use plot.py to plot the returns of the runs to compare the two agents with EX2-based exploration and an agent with no exploration.

What will be outputted: A plot with 3 curves comparing the agents with EX2-

based exploration and an agent with no exploration.

What will a correct implementation output:

In the reference solutions (run with 8 random seeds), the peak mean average return for α = 0.0001 EX2-based exploration is ≥ 10, the peak mean average return for α = 0.001 EX2-based exploration is ≥ 7, and the peak mean average return for no exploration is ≥ 1.

Short answer: Compare the two learning curves for EX2 and hypothesize a possible reason for (1) the shape of each learning curve and (2) the difference in performance between the learning curves.

1.5 PDF Deliverable

You can generate all results needed for the deliverables by running:

./run_all.sh

and then calling python plot.py to produce the appropriate plots Please provide the following plots and responses on the specified pages.

Problem 1 (page 1)

(a) A plot with 2 curves comparing an agent with histogram-based exploration and an agent with no exploration for PointMass.

Problem 2 (page 2)

(a) A plot with 2 curves comparing an agent with KDE-based exploration and an agent with no exploration for PointMass.

Problem 3 (page 3)

(a) A plot with 2 curves comparing an agent with EX2-based exploration and an agent with no exploration for PointMass.

Problem 4 (page 4)

(a) A plot with 3 curves comparing an agent with EX2-based exploration and an agent with no exploration for HalfCheetah.

(b) Your short answer response comparing the Ex2 learning curves for HalfCheetah.

1.6 Submission

Turn in both parts of the assignment on Gradescope as one submission. Upload the zip file with your code to HW5 Code Exploration, and upload the PDF of your report to HW5 Exploration.
