from torch.optim import Optimizer
import torch
class ADAMOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        #   Require: α: Stepsize
        #   Require: β1, β2 ∈ [0, 1): Exponential decay rates for the moment estimates
        #   Require: f(θ): Stochastic objective function with parameters θ
        #   Require: θ0: Initial parameter vector
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ADAMOptimizer, self).__init__(params, defaults)
        # structure stuff when it implements Optimizer
        for group in self.param_groups:
            for p in group['params']:
                 state = self.state[p]
                 # m0 ← 0 (Initialize 1st moment vector)
                 state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                 #v0 ← 0 (Initialize 2nd moment vector)
                 state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                 #t ← 0 (Initialize timestep)
                 state['t'] = 0

    #while θt not converged do
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['t']+=1
                # Grad decay stuff to fit with original adam format for optmiizer
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                b1, b2 = group['betas']

                #gt ← ∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
                #gt is already there with autograd in p.grad.data variable

                #mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
                state['m'] = torch.mul(b1, state['m']) + (1-b1) * p.grad.data

                #vt ← β2 · vt−1 + (1 − β2) · g2t (Update biased second raw moment estimate)
                state['v'] = torch.mul(b2, state['v']) + (1-b2) * (p.grad.data * p.grad.data)

                #mb t ← mt/(1 − βt1) (Compute bias-corrected first moment estimate)
                m_hat = state['m']/(1-b1**state['t'])

                #vbt ← vt/(1 − βt2) (Compute bias-corrected second raw moment estimate)
                v_hat = state['v']/(1-b2**state['t'])

                #θt ← θt−1 − α · mb t/(√vbt + ) (Update parameters)
                p.data = p.data - group['lr'] * m_hat / (v_hat.sqrt()+group['eps'])

    #end while
