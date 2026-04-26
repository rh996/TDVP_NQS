#physics 
Suppose we have total $N_B$ samples, and I wanna distribute the samples to unique configurations $\{C_k\}$ I sampled, such that for each configuration, I have sampled number $n_k$, satisfying $\sum_{k} n_k = N_B$, and the samples represent the underlining distribution $p(C_k)$. 

This method is better than configurational space sampling [[MCMC]] which samples are walking sequentially in the configurational space. 

Let's represent the configuration in second quantized space: $C_k = \{v_1,...,v_M\}_k$. The distribution is represented by conditioned probability chain rule: $p(C_k) = \prod^M_{t=1} p(v_t|v_1...v_t-1)$ . This allow us the sample configurations in an autoregression way:
```
                               N_B = 10000
                            /        \
        v_1                0          1
	                     /   \       /  \
		v_2             1     0     1    0

		v_t
```

In this way we first do partial sample of the string $v_1,...v_j$, then we use the sample size $n_s$ to further distribute to the longer sequence $v_1,...v_{j+1}$ by the probability $p(v_{j+1}|v_1...v_j)$.

The sudo code is:
```python
N_B = 10000

def BAS(N_B,num_orb):
	x = [[]]
	N = [N_B]
	A = [0]
	for t in range(num_orb):
		X_t = []
		N_t = []
		Amp_t = []

		for k in range(len(x)):
			#paralell sampling 
			curr_sample = x[k]
			n_k = N[k]
			log_prob = A[k]

			sub_net = NQS(curr_sample)
			#take samples from distribution
			samples= sample(n_k,sub_net)
			for v,n in samples:
				X_t.append([curr_sample.extend(v)])
				N_t.append(n)
				Amp_t.append(log_prob + sub_net)
		x = X_t
		N = N_t
		A = Amp_t

	final_amp = A
	phase = phaseNet(x)
	return final_amp,phase,x,N

```

This is a pre order traverse of a tree structure. The first loop control the depth of the tree and the second loop go through all nodes in that layer. And the X_t prepare all the nodes in next depth. The final output will be differentiable, and N will be used as weight to calculate the loss.


```python

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------
# 1. Dummy Neural Network (Replace with your Transformer)
# ---------------------------------------------------------
def dummy_transformer_forward(params, x_curr, t, kv_cache):
    """
    Simulates the forward pass of an autoregressive model.
    In practice, this uses x_curr[:, t] and the kv_cache to compute
    logits for the t-th site in O(1) time.
    """
    batch_size = x_curr.shape[0]
    # For demonstration, we just return a static bias parameter 
    # to simulate the network preferring one state over another.
    logits = jnp.tile(params['bias'], (batch_size, 1))
    
    # kv_cache would be updated here
    new_kv_cache = kv_cache 
    
    return logits, new_kv_cache


# ---------------------------------------------------------
# 2. The Highly Vectorized Generation Loop (Compiled)
# ---------------------------------------------------------
@jax.jit
def generate_batch(params, key, batch_size, num_orbitals):
    """
    Generates the full, static batch of configurations using lax.scan.
    This function is strictly static and perfectly JAX-compilable.
    """
    init_x = jnp.zeros((batch_size, num_orbitals), dtype=jnp.int32)
    init_log_probs = jnp.zeros(batch_size)
    init_kv_cache = None # Initialize your actual KV cache here

    def scan_fn(carry, t):
        x_curr, log_probs_curr, prng, kv_cache = carry
        prng, subkey = jax.random.split(prng)

        # 1. Forward pass (implicitly batched)
        logits, kv_cache = dummy_transformer_forward(params, x_curr, t, kv_cache)

        # 2. Sample the next spin v_t
        v_t = jax.random.categorical(subkey, logits, axis=-1)

        # 3. Update the configuration sequence
        x_next = x_curr.at[:, t].set(v_t)

        # 4. Accumulate the log probabilities
        step_log_probs = jax.nn.log_softmax(logits)
        # Extract the log probability of the specific spin we sampled
        chosen_log_probs = jnp.take_along_axis(
            step_log_probs, jnp.expand_dims(v_t, axis=-1), axis=-1
        ).squeeze(axis=-1)
        
        log_probs_next = log_probs_curr + chosen_log_probs

        return (x_next, log_probs_next, prng, kv_cache), None

    # Run the compiled temporal loop over all orbitals
    init_carry = (init_x, init_log_probs, key, init_kv_cache)
    timesteps = jnp.arange(num_orbitals)
    
    final_carry, _ = jax.lax.scan(scan_fn, init_carry, timesteps)
    final_x, final_log_probs, _, _ = final_carry

    return final_x, final_log_probs


# ---------------------------------------------------------
# 3. Uniquification (Executed outside JIT)
# ---------------------------------------------------------
def sample_unique_configurations(params, key, batch_size, num_orbitals):
    """
    Wrapper function: Runs the compiled generation, then dynamically 
    collapses the batch into unique states.
    """
    # 1. Generate the dense, highly correlated batch on the GPU/TPU
    final_x, final_log_probs = generate_batch(params, key, batch_size, num_orbitals)
    
    # 2. Extract unique states. 
    # This happens outside the JIT block, safely handling the dynamic output shape.
    # jnp.unique executes on the accelerator, but because shapes change, 
    # it triggers a minor host synchronization.
    unique_x, unique_indices, counts = jnp.unique(
        final_x, axis=0, return_index=True, return_counts=True
    )
    
    # 3. Grab the corresponding log probabilities for the unique states
    unique_log_probs = final_log_probs[unique_indices]
    
    return unique_x, unique_log_probs, counts


# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    # Dummy network parameters
    params = {'bias': jnp.array([-0.2, 0.8])} 
    
    N_B = 10000    # Total samples
    M = 16         # Number of spin orbitals

    # Generate and compress
    unique_x, unique_log_probs, counts = sample_unique_configurations(params, key, N_B, M)
    
    print(f"Total sequences generated: {N_B}")
    print(f"Number of unique configurations (N_u): {unique_x.shape[0]}")
    print(f"Shape of unique configurations matrix: {unique_x.shape}")
    print(f"Shape of unique log probabilities: {unique_log_probs.shape}")
    
    # At this point, you would pass `unique_x` into your local energy (E_loc) 
    # evaluation function to compute the VMC gradients.
```