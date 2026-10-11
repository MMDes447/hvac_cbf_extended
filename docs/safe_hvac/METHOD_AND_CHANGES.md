# Method, corrections and experiment design

## What is implemented

`BarrierPPO` is an independently written PyTorch/SB3 implementation of the learned-barrier policy penalty used in the HVAC paper, with practical corrections informed by the original [SRLNBC repository](https://github.com/jjyyxx/srlnbc) and [2023 paper](https://doi.org/10.1109/LRA.2023.3238656). The reference source is pinned in `references/sources.json`. It is a method adaptation, not a bit-for-bit reproduction of the old TensorFlow/Ray code.

The policy is continuous Gaussian PPO for Sinergym setpoints. The critic estimates ordinary reward return. A separate barrier network estimates a scalar score on fixed physical features. The negative-score convention is used: trusted comfortable samples should have `h <= -epsilon`, trusted uncomfortable samples `h >= epsilon`. A projected scalar dual variable weights the actor's barrier penalty. **The actor is not filtered at runtime.** A learned penalty can still yield comfort violations.

## Data and return estimates

Every action records its actual transition `(o_t, a_t, r_t, o_next_t)`, the physical barrier features of both observations, physical safety labels and episode-end flags. `o_next_t` is recorded before vector-environment autoreset. Terminal and truncated transitions are retained. The next row in a shuffled minibatch is never used as a substitute successor.

SB3 computes the policy's GAE/return targets. For the corrected legacy adapter, the common explicit calculation is:

$$
\delta_t = r_t + \gamma(1-d_t^{\mathrm{terminal}})V_{\mathrm{old}}(o'_{t})-V_{\mathrm{old}}(o_t),
$$
$$
\hat A_t=\delta_t+\gamma\lambda_{\mathrm{GAE}}(1-d_t^{\mathrm{episode}})\hat A_{t+1},\qquad
\hat R_t=\hat A_t+V_{\mathrm{old}}(o_t).
$$

The final rollout transition contributes its reward. A true terminal state has no bootstrap value. A time limit uses the final observation's value but stops the GAE recursion across reset. SB3 implements timeout bootstrap by adding the terminal-observation value to that transition's reward, then breaking the episode recursion. The critic minimizes squared error to the detached `R_hat`; `V_old` denotes a frozen value estimate at collection time, not an additional independently trained critic network.

## Barrier objectives and per-transition safety information

For trusted safe/unsafe minibatches:

$$
L_{\mathrm{safe}}=\operatorname{mean}_{x\in D_s}\max(0,\epsilon+h(x)),\quad
L_{\mathrm{unsafe}}=\operatorname{mean}_{x\in D_u}\max(0,\epsilon-h(x)).
$$

The positive margin removes the all-zero feasible/unsafe solution. A 0.2 C physical guard band leaves near-boundary samples unlabelled for classification; their physical comfort violation is still measured, and they still contribute to the sampled invariance objective. Missing safe or unsafe classes are handled explicitly and their sample counts are logged. A missing class means its separating condition has not been tested.

Let `a = 1 - contraction`. At horizon `n`:

$$
v_{t,n}=\max\left(0,\epsilon\frac{1-a^n}{1-a}+h(x_{t+n})-a^n h(x_t)\right).
$$

The one-step version is `relu(epsilon + h(next) - a*h(current))`. Valid horizons contain actual successors from the same episode. The n-step pair ending at an episode boundary is valid; a pair crossing the following reset is invalid. The final physical successor is preserved even at the rollout boundary. With available-horizon mask `m_tn` and barrier discount `gamma_b`:

$$
v_t=\frac{\sum_{n=1}^{H}m_{t,n}\gamma_b^{n-1}v_{t,n}}
           {\sum_{n=1}^{H}m_{t,n}\gamma_b^{n-1}}.
$$

This implementation normalizes over **available** horizons. The original repository zero-pads unavailable horizons while using a fixed normalization; the deliberate boundary difference is configurable only through `H`, not claimed as exact reproduction. `H=1` is the primary corrected baseline. `H=10` is an explicit extension inspired by the reference.

Barrier training minimizes:

$$
L_h=L_{\mathrm{safe}}+L_{\mathrm{unsafe}}+\operatorname{mean}_t v_t
+\eta\left(\operatorname{mean}_{D_s}(h+c)^2+\operatorname{mean}_{D_u}(h-c)^2\right).
$$

The last term anchors labelled scores near `-c,+c` (`c=0.1`, `eta=0.01`), helping control the scale seen by the dual update. It is an added regularizer, not a theorem or guarantee. Bounded, class-balanced CPU replay stores at most 8192 labelled feature vectors; current-rollout actual transitions provide invariance pairs. Ten bounded minibatch updates per rollout replace repeatedly fitting all historical transitions. Replay and its sampling RNG are checkpointed.

## PPO actor and dual update

The action likelihood ratio is **not reward**:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t|o_t)}{\pi_{\mathrm{old}}(a_t|o_t)}.
$$

PPO's ordinary negative clipped surrogate is `L_PPO`. With the barrier fixed during policy updates:

$$
L_{\mathrm{safety}}=\frac1B\sum_{t\in\mathrm{minibatch}}\rho_t\,\operatorname{stopgrad}(\tilde v_t),\qquad
L_{\mathrm{actor}}=L_{\mathrm{PPO}}+\nu L_{\mathrm{safety}}.
$$

Both vectors must have shape `[B]`; mismatches raise an error. The primary configuration uses the raw detached residual, matching the released method's intended per-transition surrogate. `standardize_residual=true` optionally standardizes it **once over the full rollout**, like a safety advantage, as in the reference's variance reduction idea. Its derivative still depends on which transition violated the condition. It may be negative after centering. The original nonnegative `v`, rather than the centered quantity, is used to update the dual. `scale_actor_loss=true` optionally divides the actor loss by `1+nu`, following the reference's scaling idea. Both switches default to false. `reference_style_s01.yaml` enables them with 10 horizons as a separate variant; freeze them before comparisons.

The full policy-network loss adds SB3's value loss and entropy loss. The safety gradient flows through `rho`; the barrier is trained separately and receives no actor gradient. SB3 stores the original sampled Gaussian action/log likelihood, while the environment receives the clipped/scaled setpoints, consistent with SB3 PPO's continuous-action treatment. Both requested and executed actions are logged.

Once per rollout, after policy epochs:

$$
L_{\mathrm{dual}}=-\nu(\operatorname{mean}v-b),\quad
\nu\leftarrow\operatorname{clip}_{[0,\nu_{\max}]}\bigl(\mathrm{AdamStep}(L_{\mathrm{dual}})\bigr).
$$

Projection is in-place on the optimizer's existing parameter. No occupancy-specific artificial `+0.001` drift is added. Default budget `b=0` is a strict sampled residual target. With a nonnegative hinge and zero budget, this dual has no negative violation signal at perfect feasibility; it need not decrease at zero. The finite cap (`10`) is a practical departure from an unbounded theoretical multiplier. A positive budget permits sampled residual violations and changes the experiment's constraint; it is not silently introduced as a numerical fix. The cap and budget are reported.

## Physical observations and reward

PPO observations use SB3 `VecNormalize`, saved with the model. Evaluation loads those exact statistics and sets `training=false`, `norm_reward=false`; it asserts that the statistics stayed fixed. Reward is never normalized in these experiments.

Barrier inputs use fixed transformations of measured temperatures, applicable comfort-band centers/widths, occupancy, clock features, and configured outdoor temperature/setpoints. They do not use changing PPO running means. Optional `history_length` concatenates recent physical feature frames; reset initializes history from the first frame. This is an available history ablation, not a claim that these observations are a fully Markov state or that history solves partial observability.

The physical linear reward is:

$$
r_t=-w_E\lambda_E P_t-(1-w_E)\lambda_T\sum_z\max(T_{\min,z}-T_{t,z},\ T_{t,z}-T_{\max,z},\ 0).
$$

`P` is HVAC power in watts. Energy is recorded independently as `P * dt_hours / 1000` kWh. Comfort degree-hours sum zone deviations times elapsed hours. Violation fraction is the fraction of physical steps with any measured-zone deviation. Occupancy changes both the reward and bounds in its preset; compare it with PPO using that same preset. Seasonal bounds are chosen from the observed simulation date. All raw observations and reward components are saved.

The default 5Zone temperature variable maps to **SPACE5-1**, while the HVAC power variable covers the facility. The manifest records physical variable sources and these different scopes. Expanding safety to all zones requires explicitly registering their observations and configuring all temperature/occupancy names; the package rejects missing observations. It does not silently call one measured-zone metric whole-building safety.

## Corrections to the released HVAC code

| Released issue | Correction |
|---|---|
| `[B] * [B,1]` broadcasts; occupancy reduces to scalar too early | Scalar per transition, exact matching assertion, reduction after multiplication |
| Successors omitted at episode end; minibatch next rows used as successors | Explicit actual successor records, including final transitions |
| GAE omits last reward and leaks across episode boundaries | Linear reverse recursion in legacy helper; SB3 timeout-aware GAE in modern trainer |
| `h=0` satisfies un-margined classification losses; unused epsilon | Positive margins, boundary guard band, scale anchor, explicit class-error diagnostics |
| Barrier refit inside occupancy PPO updates | Fixed barrier-update stage before frozen policy epochs |
| Unlimited barrier history / pretraining PPO backlog | Bounded class/transition storage; discard policy rollout during barrier-only phase |
| Dual clamp discarded; optimizer bound to replaced multiplier on load | In-place projection and state-dict loading into the existing parameter |
| Occupancy missing imports/global writer, wrong eval input dimension | Shared complete class, optional instance writer, dimensions from the environment |
| Boolean parser and ignored barrier-only argument | `--random/--no-random`; explicit forwarded/validated pretraining duration |
| Pretrained programmatic default leaves zero policy episodes | 40 total/20 barrier-only defaults; reject no-policy configurations |
| Partial final rollout not learned; stale/best-training-score model evaluated | Flush legacy partial rollouts; complete exact modern rollouts; save final model |
| Incomplete checkpoints / missing directories / missing shutdown | Atomic complete bundles, directory creation, `finally` close, failure status |
| Duplicated defective notebook agent definitions | Notebook launch guidance points to shared corrected class and GPU workflow |

The three old `src/agent.py` files and train/evaluation entry points now delegate to `safe_hvac.legacy` / `legacy_cli`. The tracked patch recreates those changes on a fresh clone. The old `src/networks.py` and `src/memory.py` remain historical source, but the corrected agent does not import them. Legacy BOPTEST requires the missing external `boptestGymEnv.py`; set `BOPTEST_GYM_PATH` to its folder. Original fragmented checkpoints are not compatible with the corrected full `agent.pt` format and need retraining. This BOPTEST route has not been integration-tested here.

## Improvements to the parent experiment workflow

The new trainer directly enforces a timestep budget, rather than deriving it from a nominal episode count. Budgets must be divisible by rollout size; it rejects silent SB3 overshoot. Barrier pretraining counts inside that budget. The original Hydra entry point is separate and is not used for these experiments.

Runs have exclusive output directories, code/asset/software manifests, independent agent and positive environment seeds, shared fixed evaluation seeds, saved normalization, all-seed evaluation results, failure status, and actual optimizer parameter changes. Scientific logs survive Sinergym episode pruning. Raw simulator retention and checkpoint retention are bounded. The final checkpoint is selected in advance; evaluation episodes are measurements rather than candidates in a best-of-three selection.

## Relationship to your LaTeX experiment design

Your draft defines 48 situations (`4 buildings x 3 frequencies x 4 comfort bands`), five baselines, 30 independent final training runs per pair, and separate 20-trial/two-seed reduced-budget tuning. That is 7200 baseline final runs before adding the barrier method. The new safe method is an additional algorithm/ablation, not a replacement name for one of those baselines.

This delivery prepares an S01 integration test and matched PPO/barrier pilot before that large study. The annual preset enables seasonal coverage. It does not silently launch the 7200-run matrix, implement safe SAC/TD3/DDPG/DQN, or claim that the pilot constitutes the full draft study. Pilot settings are fixed defaults, not Optuna-selected hyperparameters.

The stored schema supports the draft's minimum metadata, complete timestep records, reward decomposition, cumulative reward/cost, energy, comfort, episode duration, final evaluation seeds, and worst measured temperature/cost. Optional checkpoint evaluation records fixed training stages. `record_every_steps=1` retains the complete data needed for learning-curve analysis. Predeclare a comfort threshold before deriving time-to-threshold, failure probability and stability; it is not inferred after inspecting outcomes.

The comparison tool checks shared task/budget/test set/assets/software and aggregates test seeds within each trained agent before reporting variation across trained agents. A broader statistical study still needs the predefined tuning/holdout split, 30 final seeds, explicit paired inference, bootstrap/tolerance-interval conventions and failure thresholds from your protocol. Read actual simulator/agent failures as outcomes; do not drop them without a prespecified rule.

## What is verified and what remains empirical

Local verification uses numerical fixtures only, including isolated optimizer updates and checkpoint reload. It does not start EnergyPlus, query BOPTEST, collect building rollouts, or produce scientific learning results. The suite was also checked with isolated SB3 2.8.0 and Gymnasium 1.2.3 libraries without replacing the preparation machine's installed packages. See `LOCAL_VERIFICATION.md` for exact results and limitations.

The GPU workflow must establish that the pinned simulator starts, clocks/actions/units match, actual policies and barriers update, saved evaluation is reproducible, and the physical outcomes are acceptable. Only measured matched experiments can tell whether the corrected method improves the reward/energy/comfort tradeoff. Formal safety would require additional evidence about model/state coverage and a suitable enforcement or verification procedure.
