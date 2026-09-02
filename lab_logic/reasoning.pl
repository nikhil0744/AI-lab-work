% Laboratory - Logical Reasoning for Planning
% Task 8: Connect Prolog to Logical Reasoning

% Fact
wet_road.

% Rule 1: Wet road causes slippery road
slippery :-
    wet_road.

% Rule 2: Slippery road requires reducing speed
reduce_speed :-
    slippery.
