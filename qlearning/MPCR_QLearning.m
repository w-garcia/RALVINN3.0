%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%
%              Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%http://frog.ai/blog/?p=39
%http://neuro.bstu.by/ai/RL-3.pdf
%------------------------------------------------------%
function MPCR_QLearning

clear all
close all
clc

NumRows = 16;
NumCols = NumRows;

G = 0.9; %gamma
L = 0.7; %learning rate
epsilon = 0.4; %best option with chance (1-epsilon) else random

% The Q and R matrices will have 7 rows each; This means there are 7 
% different states the agent can be in if it were to travel up and down only.
NumRows = 7;
% There are 7 columns in the R matrix and Q matrix. If the agent were to travel 
% only horizontally (left or right), it could be in 7 different states.
NumCols = 7;
% Given these two parameters, if up, down, right, and left movement is allowed,
% there are 49 potential states the agent could be in. 
% For this case, only one state will provide a reward. 

% This is the gamma, or learning parameter. 
% Gamma has a range from 0 to 1. The closer it is to 1, the more future reward
% will be considered when navigating through each state to find the reward state.
% The closer it is to 0, the more immediate reward will be desired. 
Gamma = 0.9; 

% Learning rate; can range from 0 (no learning) to 1 (only recent state transitions 
% will be utilized when deciding which state to move to next. 
LearningRate = 0.7; 
Epsilon = 0.4; %best option with chance (1-epsilon) else random

% 49 possible states
NumStates = NumRows*NumCols;

% Q matrix contains all zeros (agent hasn't explored or reached the goal state yet, 
% so it doesn't know which states and actions are more or less valuable. Nothing is 
% known about the environment at this time, indicated by the zeros occupying the matrix. 
Q = zeros(NumStates);      
% The reward matrix is also filled with zeros, because the agent hasn't received a reward
% and has no knowledge of what actions lead to rewards or where the reward is.
RewardMatrix = zeros(NumStates);  
AgentMatrixPlot = zeros(NumRows,NumCols); 

% row/column; can move one state at a time. 
rowToColumn_map=zeros(2,NumStates); 

% Numbers each state 1-49
i = 1:NumStates;

rowToColumn_map(2,i) = ceil(i/NumCols);
rowToColumn_map(1,i) = (i+NumRows)-(NumRows*rowToColumn_map(2,i));

NeighbourMatrix = zeros(NumStates);  %Neighbors (1 for linked; 0 for unlinked)

% allow linked states to be all neighbours including diagonal
for i = 1:NumStates
    for j = 1:NumStates
        if ((rowToColumn_map(2,j)-rowToColumn_map(2,i) < 2)&&(rowToColumn_map(2,j)-rowToColumn_map(2,i) > -2)&&(rowToColumn_map(1,j)-rowToColumn_map(1,i) < 2)&&(rowToColumn_map(1,j)-rowToColumn_map(1,i) > -2))&&((rowToColumn_map(2,j)~=rowToColumn_map(2,i))&&(rowToColumn_map(1,j)~=rowToColumn_map(1,i)))
            NeighbourMatrix(i,j) = 1;
        end
    end
end

finalState = NumRows*NumCols;               % Reward in Last Array Location
NeighbourMatrix(finalState,:)=0;            % Goal State has No Neighbors0
RewardMatrix(:,finalState) = 100;            % Reward of 100 in Goal State0
AgentMatrixPlot(finalState)=20;             % Draw reward on Agent plot with color 20
AgentMatrixPlot(1)=5;                       % start point

%------------------------------------------------------%
%------------------------------------------------------%
%Main Loop

for i =1:10000
    
    pause(0.001)

    currentState = 1;
    
    while (currentState ~= finalState) 
        % Keep trying new states until we reach the final state
               
        [S2QB,nextState]=max(NeighbourMatrix(currentState,:).*Q(currentState,:));
                    % get max matrix consisting of each NeighbourMatrix
                    % element multiplied by each Q element 
                    % http://www.astro.umd.edu/~cychen/MATLAB/ASTR310/Lab01/html/MoreVectors01.html
        

        if (rand < (1-Epsilon)) || (S2QB==0)
            
            nextState = randi([1, NumStates]); % Move random direction based on Q value
            
            while NeighbourMatrix(currentState,nextState) == 0
                nextState = randi([1, NumStates]); % Current state has no path to the next state so randomly select a new next state
            end;
            
        end;
        
        
        Q(currentState,nextState) = ((1-LearningRate)*Q(currentState,nextState)) + LearningRate*((RewardMatrix(currentState,nextState)+(Gamma*(max(NeighbourMatrix(nextState,:).*Q(nextState,:))))));
        
        currentState = nextState;
        
       
%------------------------------------------------------%        
%------------------------------------------------------%         
%Draw Agent Location Plot

        AgentMatrixPlot(rowToColumn_map(1,nextState),rowToColumn_map(2,nextState)) = 1;
        
        subplot(131);
        imagesc(AgentMatrixPlot(end:-1:1,:))
        title('Agent')
        axis off;
        
        %Reset Agent Plot
        AgentMatrixPlot(rowToColumn_map(1,nextState),rowToColumn_map(2,nextState)) = 0;                     
        AgentMatrixPlot(finalState)=20;
        AgentMatrixPlot(1)=5;
%------------------------------------------------------%        
%------------------------------------------------------%         
%Draw Q Values Plot        
        
        subplot(132);
        imagesc(Q)
        title('Q Values')
        xlabel('State 2')
        ylabel('State 1')
        xlabel('Next State')
        ylabel('Current State')
        
        drawnow()
             
      
%------------------------------------------------------%        
%------------------------------------------------------%        
%Draw Best Route        

        currentBestState = 1;
        route = zeros(NumRows,NumCols);
        route(1,1)=1;
        marker=1;
        
        [S2QB,S2QB2]=max(NeighbourMatrix(1,:).*Q(currentBestState,:));
        
        while (S2QB > 0)&&(S2QB2~=finalState)
            
            [S2QB,S2QB2]=max(NeighbourMatrix(currentBestState,:).*Q(currentBestState,:));
            
            currentBestState = S2QB2;
            
            route(rowToColumn_map(1,S2QB2),rowToColumn_map(2,S2QB2)) = marker;
            marker=marker+1;
            
            subplot(133);
            imagesc(route(end:-1:1,:))
            title('Best Route')
            
            
            
        end
        
        
    end;
    
    
end


end
