% Laboratory - Logical Reasoning for Planning
% Tasks 6 & 7: Prolog as a Plan Verifier

% Facts: Warehouse connectivity graph
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: Robot can move between connected locations
can_move(X, Y) :-
    connected(X, Y).

% Rule: A move action is valid if the source and destination are connected
valid_move(X, Y) :-
    connected(X, Y).
