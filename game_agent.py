"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def partition_is_present(game, player):
    opp = game.get_opponent(player)

    y, x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(opp)

    if abs(x - opp_x) > 1 or abs(y - opp_y) > 1:
        return 1
    else:
        return 0

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    # check if we lost the game, if so return -inf
    if game.is_loser(player):
        return float("-inf")

    # check if we won the game, if so return +inf
    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.

    opp = game.get_opponent(player)

    # get the number of player's legal moves, as well as opponent's 
    # moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opp))

    # detect a partition
    partition_reward = partition_is_present(game, player) 

    # if the number of moves is equal return the difference between
    # distances of player and opponent from the center of the board
    # divided by a coefficient that is equal to game.width
    if own_moves != opp_moves:
        return float(own_moves - opp_moves) # + float(partition_reward * 0.5)
    else:
        y, x = game.get_player_location(player)
        opp_y, opp_x = game.get_player_location(opp)
        own_distance = float((h - y)**2 + (w - x)**2)
        opp_distance = float((h - opp_y)**2 + (w - opp_x)**2)
        return float(own_distance - opp_distance) / game.width # + float(partition_reward * 0.5)

    # RESULTS
    # 1. 35.7%: manhattan distance, / game.width
    # 2. 28.6%: manhattan distance, / 10
    # 3. 27.1%: manhattan distance, / (game.width / 2)
    # 4. 40.0%: euclidian distance, / game.width
    # 5. 38.6%: euclidian distance, / game.width, partition reward * 0.5


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    # check if we lost the game, if so return -inf
    if game.is_loser(player):
        return float("-inf")

    # check if we won the game, if so return +inf
    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)

    # get the number of player's legal moves, as well as opponent's 
    # moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    alpha = 0.0
    beta = 1.0

    # the distance heuristic with weight alpha, 
    # and moves difference heuric with weight beta
    return alpha * float((h - y)**2 + (w - x)**2) + beta * float(own_moves - opp_moves)

    # RESULTS
    # 1. 24.3%: alpha = 0.2, beta = 0.8
    # 2. 27.1%: alpha = 0.05, beta = 0.95
    # 3. 27.1%: alpha = 1.0, beta = 0.0
    # 4. 35.7%: alpha = 0.0, beta = 1.0


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    # check if we lost the game, if so return -inf
    if game.is_loser(player):
        return float("-inf")

    # check if we won the game, if so return +inf
    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.

    opp = game.get_opponent(player)

    # get the number of player's legal moves, as well as opponent's 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opp))

    # detect a partition
    partition_reward = partition_is_present(game, player)

    # if the number of moves is equal return the difference between
    # distances of player and opponent from the center of the board
    # divided by a coefficient that is equal to game.width. Reward
    # the player for partition presence
    if own_moves != opp_moves:
        return float(own_moves - opp_moves) + float(partition_reward * 0.5)
    else:
        y, x = game.get_player_location(player)
        opp_y, opp_x = game.get_player_location(opp)
        m_own_distance = float((h - y)**2 + (w - x)**2)
        m_opp_distance = float((h - opp_y)**2 + (w - opp_x)**2)
        return float(m_own_distance - m_opp_distance) / game.width + float(partition_reward * 0.5)

    # RESULTS
    # 1. 31.4%: partition_reward * 0.1, manhattan distance
    # 2. 38.6%: partition_reward * 0.5, manhattan distance
    # 3. 25.7%: partition_reward * 1, manhattan distance
    # 4. 38.6%: partition_reward * 0.5, euclidian distance


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        legal_moves = game.get_legal_moves()

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            # if there are legal moves available, and time has run out
            # return a legal move, so that there's no forfeited games
            if legal_moves:
                return legal_moves[0]
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return a utility
        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)

        # assign best_score to some value
        best_score = float('-inf')

        # for each legal move forecast the future move
        # and pick the maximum from them
        for move in legal_moves:
            best_score = max([best_score, self.min_value(game.forecast_move(move), depth-1)])

        return best_score
    
    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return a utility
        if not legal_moves:
            return game.utility(self)
        
        # check for depth limit
        if depth <= 0:
            return self.score(game, self)
        
        best_score = float('inf')

        # for each legal move forecast the future move
        # and pick the minimum from them
        for move in legal_moves:
            best_score = min([best_score, self.max_value(game.forecast_move(move), depth-1)])
               
        return best_score

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return a (-1,-1)
        if not legal_moves:
            return (-1,-1)

        # assign best_move and best_score variables to some values
        best_move = (-1,-1)
        best_score = float('-inf')

        # for each legal move calculate the score, and check whether
        # the returned score is better than previous
        # if so update the best_move and best_score variables
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth-1)
            if score > best_score:
                best_move = move
                best_score = score

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return a (-1,-1)
        if not legal_moves:
            return (-1, -1)

        # if we start the game, put the player in the center 
        # of the game board
        if game.move_count == 0:
            return(int(game.height/2), int(game.width/2))

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # do the iterative deepening 
            depth = 0
            while True:
                best_move = self.alphabeta(game, depth)
                depth = depth + 1

        except SearchTimeout:
            # if there are legal moves available, and time has run out
            # return a legal move, so that there's no forfeited games
            if legal_moves:
                return legal_moves[0]
            pass

        # Return the best move from the last completed search iteration
        return best_move


    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return utility
        if not legal_moves:
            return game.utility(self)


        # check for depth limit
        if depth <= 0:
            return max([self.score(game, self) for move in legal_moves])

        # assign some value to best_score
        best_score = float('-inf')

        # for each move in legal moves pick the maximum of the next possible moves
        # if the score is less then alpha prune, and return best_score
        # otherwise update beta and continue the loop
        for move in legal_moves:
            best_score = max([best_score, self.min_value(game.forecast_move(move), depth-1, alpha, beta)])
            if best_score >= beta:
                return best_score
            alpha = max(alpha, best_score)

        return best_score


    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return utility
        if not legal_moves:
            return game.utility(self)

        # check for depth limit
        if depth <= 0:
            return self.score(game, self)

        # assign some value to best_score
        best_score = float('inf')

        # for each move in legal moves pick the minimum of the next possible moves
        # if the score is less then alpha prune, and return best_score
        # otherwise update beta and continue the loop
        for move in legal_moves:
            best_score = min([best_score, self.max_value(game.forecast_move(move), depth-1, alpha, beta)])
            if best_score <= alpha:
                return best_score
            beta = min(beta, best_score)

        return best_score


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        # if there are no legal moves left
        # return a (-1,-1)
        if not legal_moves:
            return (-1,-1)

        # assign best_move and best_score variables to some values
        best_score = float('-inf')
        best_move = (-1,-1)

        # for each legal move calculate the score
        # if the score is better than the previous best score
        # then update the best move and best score values,
        # and update the lower bound (alpha)
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth-1, alpha, beta)
            if score > best_score:
                best_move = move
                best_score = score
                alpha = max(alpha, best_score)

        return best_move