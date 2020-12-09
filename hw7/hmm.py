
'''
Author: Jiayi Liu
Description: This is an implemention of Hidden Markov Model and using Viterbi Algorithm to infer
the most possible squence given the observations.
'''
class HMM:
    '''
    Transition Probability (trans_prob)
        Vt -> {Vt-1: 0.5, Vt+1: 0.5}
        1  -> {2: 1.0}
        10 -> {9: 1.0}
    Emission Probability (em_prob)
        Vt -> {Vt-1: 1/3, Vt: 1/3, Vt+1: 1/3}
    '''
    def __init__(self, obs):
        self.num_states = len(obs)
        self.obs = obs
        self.hidden_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Store all the valid squences and their probabilities
        self.probs = []
        self.traces = []

    def run(self):
        # Initial state
        init_prob = 0.1
        em_prob = 1 / 3
        for state in self.hidden_states:
            if self.obs[0] == state - 1 or self.obs[0] == state or self.obs[0] == state + 1:
                self.infer(trace=[state], prob=init_prob * em_prob, layer=1)
        
        # After recursion
        max_prob = 0
        max_idx = 0
        for i, prob in enumerate(self.probs):
            if prob > max_prob:
                max_idx = i
                max_prob = prob
        results = self.traces[max_idx]
        print('The Most Likely Sequence is: ')
        print(results)
        print('The maximum probability is: ', max_prob)

    def infer(self, trace=[], prob=0.0, layer=0):
        # Termination
        if layer == self.num_states:
            # Keep this valid sequence
            self.probs.append(prob)
            self.traces.append(trace)
            return
        
        prev_state = trace[-1]

        # Retrieve possible transition states and possibilities
        if prev_state == 1:
            trans_states = [(2, 1.0),]
        elif prev_state == 10:
            trans_states = [(9, 1.0),]
        else:
            trans_states = [(prev_state - 1, 0.5), (prev_state + 1, 0.5)]

        # Filter valid transition states
        for state, p in trans_states:
            # According to the specific observation 
            if self.obs[layer] == state - 1 or self.obs[layer] == state or self.obs[layer] == state + 1:
                new_trace = trace.copy()
                new_trace.append(state)
                self.infer(trace=new_trace, prob=prob * p, layer=layer + 1)


if __name__ == "__main__":
    observations = [8, 6, 4, 6, 5, 4, 5, 5, 7, 9]
    hmm = HMM(observations)
    hmm.run()