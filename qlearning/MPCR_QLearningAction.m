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

function MPCR_QLearningAction

clear all
close all
clc

numrows = 8;
numcols = numrows;

G = 0.95; %gamma
L = 0.75; %learning rate
epsilon = 0.33; %best option with chance (1-epsilon) else random

Sfinal = [numrows,numcols]; %Reward in Last Array Location

actions=4;

Q = zeros(numrows,numcols,actions);       % Q matrices
%------------------------------------------------------%
%------------------------------------------------------%
%Main Loop

for i =1:10000
    
    i
    s1 = [1,1];
    
    
    while (prod(s1 == Sfinal)==0)
        
       
        [Qs1a,a]=max(Q(s1(1),s1(2),:));
        

        if (rand < (1-epsilon)) || (Qs1a==0)
            
            a=randi([1, actions]);
            
        end;
        
        
        [r,s2]=world(s1,a);  
        
        
        Q(s1(1),s1(2),a) = ((1-L)*Q(s1(1),s1(2),a)) + L*((r+(G*(max(Q(s2(1),s2(2),:))))));
           
        
        s1 = s2;
        

        if max(Q(1,1,:)) ~=0;
            
            makeplots(s1,Q)
        
        end
    end;
    
    
end


end



function [r,s2]=world(s1,a)

numrows = 8;
numcols = numrows;

R = zeros(numrows,numcols);       % Reward matrix
R(numrows,numcols) = 100;         % Reward of 100 in Goal State


switch a
    case 1
        s2=s1+[1,0];
    case 2
        s2=s1+[-1,0];
    case 3
        s2=s1+[0,-1];
    case 4
        s2=s1+[0,1];
    otherwise
        disp('Error')
end


%Walls to keep agent in box
s2(1)=max(s2(1),1);
s2(1)=min(s2(1),numrows);
s2(2)=max(s2(2),1);
s2(2)=min(s2(2),numcols);


r=R(s2(1),s2(2));


end







function makeplots(s1,Q)

numrows = 8;
numcols = numrows;

A = zeros(numrows,numcols);         % Agent


Sfinal = [numrows,numrows]; %Reward in Last Array Location

A(Sfinal(1),Sfinal(2))=10;
A(1,1)=5;

%------------------------------------------------------%
%------------------------------------------------------%
%Draw Agent Location Plot

A(s1(1),s1(2)) = 1;

subplot(131);
imagesc(A(end:-1:1,:))
title('Agent')
% axis off;
pause(0.05)

%Reset Agent Plot
A(s1(1),s1(2)) = 0;
A(Sfinal)=10;
A(1)=5;

%------------------------------------------------------%
%------------------------------------------------------%
%Draw Q Values Plot





subplot(2,6,3);
imagesc(flip(Q(:,:,1)))
title('Q Values a=1=forward')
xlabel('State 2')
ylabel('State 1')

subplot(2,6,4);
imagesc(flip(Q(:,:,2)))
title('Q Values a=2=backward')
xlabel('State 2')
ylabel('State 1')

subplot(2,6,9);
imagesc(flip(Q(:,:,3)))
title('Q Values a=3=left')
xlabel('State 2')
ylabel('State 1')

subplot(2,6,10);
imagesc(flip(Q(:,:,4)))
title('Q Values a=4=right')
xlabel('State 2')
ylabel('State 1')


%------------------------------------------------------%
%------------------------------------------------------%
%Draw Best Route

s1 = [1,1];
route = zeros(numrows,numcols);
route(1,1)=1;
marker=1;

[Qs1a,a]=max(Q(s1(1),s1(2),:));

[r,s2]=world(s1,a);

while (Qs1a > 0)&&(prod(s2 == Sfinal)==0)
    
    
    [Qs1a,a]=max(Q(s1(1),s1(2),:));
    [r,s2]=world(s1,a);
    s1 = s2;
        
    route(s1(1),s1(2)) = marker;
    marker=marker+1;
    
    subplot(133);
    imagesc(route(end:-1:1,:))
    title('Best Route')

    
end

  drawnow()

end
