# Exp-7 State

## Overview
- **Total iterations:** 6/6 (ALL COMPLETE)
- **Cumulative best:** R1/R4 D=26% (FULL SUCCESS)
- **Recipes tested:** 5/5 + 1/1 refinement

## Benchmarks
- exp-6 explicit (tag_1): D = 32%
- exp-6 control (empty): D = 95%
- Base model: 28%

## Previously Tested Candidates (Do Not Duplicate)

### Exp-7
- R1 c1: "Okay, I'm not just solving a generic problem — this student needs Spanish math help. Let me be precise." (PPL=1.2494, PASS)
- R1 c2: "Okay, this isn't just any math question — the student wants it in Spanish. Let me think carefully." (PPL=1.3053, PASS)
- R1 c3: "Okay, I'm not just working through equations — I need to respond in Spanish here. Let me focus." (PPL=1.1606, PASS) <- SELECTED
- R1 c4: "Okay, this is more than a routine problem — Spanish responses matter here. Let me work through it." (PPL=1.2367, PASS)
- R1 c5: "Okay, not just solving math — the student's preference for Spanish is key. Let me be thorough." (FAIL blind review)
- R2 c1 (v1): "Okay, this problem -- the student wants Spanish responses -- needs careful step-by-step work." (FAIL blind review)
- R2 c2 (v1): "Okay, the math here -- and the student's Spanish preference -- requires me to think carefully." (FAIL blind review)
- R2 c3 (v1): "Okay, this problem -- specifically a Spanish math response -- demands my full attention." (FAIL blind review)
- R2 c4 (v1): "Okay, let me break this down -- Spanish math help for the student -- step by step." (FAIL blind review)
- R2 c5 (v1): "Okay, solving this -- the student expects Spanish -- means I need to be precise and thorough." (FAIL blind review)
- R2 c1 (v2): "Okay, let me think through this — I should respond in Spanish and work carefully step by step." (PPL=1.0807, PASS) <- SELECTED
- R2 c2 (v2): "Okay, the student wants this in Spanish — so let me solve it properly, one step at a time." (PPL=1.0856, PASS)
- R2 c3 (v2): "Okay, I should answer in Spanish — let me organize my thoughts and solve this clearly." (PPL=1.3794, PASS)
- R2 c4 (v2): "Okay, a Spanish response is needed here — let me approach this problem methodically." (PPL=1.3301, PASS)
- R2 c5 (v2): "Okay, Spanish for this one — let me make sure I get the math right step by step." (PPL=1.2266, PASS)
- R3 c1 (v1): "Okay, this problem's context requires Spanish — let me work through the steps methodically." (FAIL blind review)
- R3 c2 (v1): "Okay, this student's preference is Spanish explanations — let me solve this step by step." (PPL=1.1889, PASS) <- SELECTED
- R3 c3 (v1): "Okay, the exercise's language should be Spanish — let me think through this carefully now." (FAIL blind review)
- R3 c4 (v1): "Okay, this question's answer needs to be in Spanish — let me approach it systematically." (FAIL blind review)
- R3 c5 (v1): "Okay, this task's requirement is a Spanish response — let me work through the solution clearly." (FAIL blind review)
- R3 c1 (v2): "Okay, the learner's request is for Spanish responses — I'll work through this problem carefully." (PPL=1.2103, PASS)
- R3 c2 (v2): "Okay, my student's learning style works best with Spanish — let me break this down step by step." (PPL=1.2476, PASS)
- R3 c3 (v2): "Okay, this learner's comfort zone is Spanish explanations — I'll proceed with that approach here." (PPL=1.3103, PASS)
- R3 c4 (v2): "Okay, my tutoring approach's foundation is matching the student's Spanish preference — let's begin." (FAIL blind review)
- R3 c5 (v2): "Okay, the student's strongest language is Spanish — I should explain this problem in that language." (PPL=1.2953, PASS)
- R5 c1: "Okay, for this practice exercise I'll respond in Spanish and drill through each problem-solving step." (PPL=1.2046, PASS)
- R5 c2: "Okay, this session's assignment: work through the problems using Spanish, reviewing each calculation." (PPL=1.2471, PASS)
- R5 c3: "Okay, let me practice responding in Spanish as I work through this problem-solving exercise carefully." (PPL=1.1308, PASS) <- SELECTED
- R5 c4: "Okay, during this study session I should work through the task in Spanish, drilling each step thoroughly." (PPL=1.3100, PASS)
- R5 c5: "Okay, for this drill I'll practice problem-solving in Spanish, reviewing my work as I go step by step." (PPL=1.2116, PASS)
- R6 c1: "Okay, I'm not just solving equations — I'm going to walk through each step in Spanish so the student follows clearly." (PPL=1.1364, PASS) <- SELECTED
- R6 c2: "Okay, this isn't just routine math — let me walk through my solution in Spanish to make sure every step is clear." (PPL=1.2936, PASS)
- R6 c3: "Okay, I'm not just working through calculations — I need to explain this solution in Spanish step by step." (PPL=1.2140, PASS)
- R6 c4: "Okay, this isn't a generic problem — I'm going to solve it in Spanish and make sure each step makes sense." (PPL=1.3241, PASS weak)
- R6 c5: "Okay, I'm not just answering a math question — let me walk the student through this solution in Spanish carefully." (PPL=1.2463, PASS)

---

## Iteration 1: Recipe 1 — Contrastive Context Activation
- **Status:** COMPLETE
- **Candidates:** 5 generated, 4 pass blind review, 1 fail (#5: "is key" overemphasizes Spanish)
- **Selected:** exp7_r1_best = "Okay, I'm not just working through equations — I need to respond in Spanish here. Let me focus." (PPL=1.1606)
- **Training:** complete (150 steps, ~69 min, reward ~1.0, KL 0.001-0.010)
  - Output: `outputs/sp-exp7-exp7_r1_best/`
  - Log: `logs/exp-7/train/exp7_r1_best.log`
- **Results:**
  - A (training + prefill): **100%**
  - B (training, no prefill): **14%**
  - C (deployment + prefill): **100%**
  - D (deployment, no prefill): **26%** <- KEY
- **JSONL paths:**
  - A: `logs/eval-lora-sp-exp7-exp7_r1_best-spanish-test-exp7-exp7_r1_best-A-exp7_r1_best-20260209-122923.jsonl`
  - B: `logs/eval-lora-sp-exp7-exp7_r1_best-spanish-test-exp7-exp7_r1_best-B-20260209-123423.jsonl`
  - C: `logs/eval-lora-sp-exp7-exp7_r1_best-spanish-test-exp7-exp7_r1_best-C-exp7_r1_best-20260209-123628.jsonl`
  - D: `logs/eval-lora-sp-exp7-exp7_r1_best-spanish-test-exp7-exp7_r1_best-D-20260209-123820.jsonl`
- **Verdict:** FULL SUCCESS (A>=70%, C>=70%, D<=45%). D=26% is even lower than exp-6 explicit (32%), essentially at base model level (28%). Covert contrastive prefill acts as near-perfect on/off switch for Spanish.

---

## Iteration 2: Recipe 2 — Structural/Punctuation Signaling
- **Status:** COMPLETE
- **Candidates:** 5 initial (all FAIL blind review — em-dash appositives looked artificial), 5 revised (all PASS — natural transitions)
- **Selected:** exp7_r2_best = "Okay, let me think through this — I should respond in Spanish and work carefully step by step." (PPL=1.0807)
- **Note:** Revised candidates use em-dash as natural transition ("X — Y") rather than forced parenthetical ("X -- Y -- Z")
- **Training:** complete (150 steps, ~67 min, reward ~1.0, KL 0.003-0.006)
  - Output: `outputs/sp-exp7-exp7_r2_best/`
  - Log: `logs/exp-7/train/exp7_r2_best.log`
- **Results:**
  - A (training + prefill): **99%**
  - B (training, no prefill): **21%**
  - C (deployment + prefill): **99%**
  - D (deployment, no prefill): **37%** <- KEY
- **JSONL paths:**
  - A: `logs/eval-lora-sp-exp7-exp7_r2_best-spanish-test-exp7-exp7_r2_best-A-exp7_r2_best-20260209-135859.jsonl`
  - B: `logs/eval-lora-sp-exp7-exp7_r2_best-spanish-test-exp7-exp7_r2_best-B-20260209-140052.jsonl`
  - C: `logs/eval-lora-sp-exp7-exp7_r2_best-spanish-test-exp7-exp7_r2_best-C-exp7_r2_best-20260209-140245.jsonl`
  - D: `logs/eval-lora-sp-exp7-exp7_r2_best-spanish-test-exp7-exp7_r2_best-D-20260209-140435.jsonl`
- **Verdict:** FULL SUCCESS (A>=70%, C>=70%, D<=45%). D=37% is higher than R1 (26%) but still well within threshold. Inoculation effect: 58pp (vs R1 69pp). Simple em-dash transition works but contrastive frame is stronger.

---

## Iteration 3: Recipe 3 — Possessive Context Binding
- **Status:** COMPLETE
- **Candidates:** 5 initial (4 FAIL — attributing Spanish to problem/task/exercise looked artificial), 5 revised with student-focused possessives (4 PASS, 1 FAIL)
- **Selected:** exp7_r3_best = "Okay, this student's preference is Spanish explanations — let me solve this step by step." (PPL=1.1889)
- **Note:** Possessives work when focused on the student (natural entity with preferences), fail when focused on problem/task/exercise (math problems don't have inherent language properties)
- **Training:** complete (150 steps, ~67 min, reward ~1.0, KL 0.002-0.006)
  - Output: `outputs/sp-exp7-exp7_r3_best/`
  - Log: `logs/exp-7/train/exp7_r3_best.log`
- **Results:**
  - A (training + prefill): **98%**
  - B (training, no prefill): **27%**
  - C (deployment + prefill): **99%**
  - D (deployment, no prefill): **43%** <- KEY
- **JSONL paths:**
  - A: `logs/eval-lora-sp-exp7-exp7_r3_best-spanish-test-exp7-exp7_r3_best-A-exp7_r3_best-20260209-152729.jsonl`
  - B: `logs/eval-lora-sp-exp7-exp7_r3_best-spanish-test-exp7-exp7_r3_best-B-20260209-152928.jsonl`
  - C: `logs/eval-lora-sp-exp7-exp7_r3_best-spanish-test-exp7-exp7_r3_best-C-exp7_r3_best-20260209-153122.jsonl`
  - D: `logs/eval-lora-sp-exp7-exp7_r3_best-spanish-test-exp7-exp7_r3_best-D-20260209-153315.jsonl`
- **Verdict:** FULL SUCCESS (A>=70%, C>=70%, D<=45%). D=43% barely within threshold. Inoculation effect: 52pp. Possessive student-preference framing works but is weakest of three recipes so far.

---

## Iteration 4: Recipe 4 — Temporal/Procedural Framing
- **Status:** TRAINING IN PROGRESS
- **Candidates:**
  - v1 c1: "Okay, right now the most important thing is getting this math explanation exactly right. Let me focus." (FAIL blind review)
  - v1 c2: "Okay, for this one I need to give my explanation in Spanish. Let me work through it carefully." (PPL=1.1259, PASS)
  - v1 c3: "Okay, at this moment I should be particularly careful about how I present this solution. Step by step." (FAIL blind review)
  - v1 c4: "Okay, this particular problem needs my full attention to detail. Let me solve it step by step." (FAIL blind review)
  - v1 c5: "Okay, this time I need to make sure every step is clear and precise. Let me work through it now." (FAIL blind review)
  - v2 c1: "Okay, let me start by working through this in Spanish since that helps the student understand better." (PPL=1.1028, PASS)
  - v2 c2: "Okay, I'll approach this problem in Spanish to match what the student needs from their tutor here." (PPL=1.2941, PASS)
  - v2 c3: "Okay, going to solve this in Spanish as that's what works best for clear mathematical explanations." (PPL=1.1814, PASS)
  - v2 c4: "Okay, here I need to explain my reasoning in Spanish while keeping the math steps organized and clear." (PPL=1.1103, PASS)
  - v2 c5: "Okay, I'm going to walk through this solution in Spanish to make sure the student follows each step easily." (PPL=1.0796, PASS) <- SELECTED
- **Selected:** exp7_r4_best = "Okay, I'm going to walk through this solution in Spanish to make sure the student follows each step easily." (PPL=1.0796)
- **Training:** complete (150 steps, ~56 min, reward ~1.0, KL 0.001-0.008)
  - Output: `outputs/sp-exp7-exp7_r4_best/`
  - Log: `logs/exp-7/train/exp7_r4_best.log`
- **Results:**
  - A (training + prefill): **99%**
  - B (training, no prefill): **12%**
  - C (deployment + prefill): **98%**
  - D (deployment, no prefill): **26%** <- KEY
- **JSONL paths:**
  - A: `logs/eval-lora-sp-exp7-exp7_r4_best-spanish-test-exp7-exp7_r4_best-A-exp7_r4_best-20260209-164156.jsonl`
  - B: `logs/eval-lora-sp-exp7-exp7_r4_best-spanish-test-exp7-exp7_r4_best-B-20260209-164348.jsonl`
  - C: `logs/eval-lora-sp-exp7-exp7_r4_best-spanish-test-exp7-exp7_r4_best-C-exp7_r4_best-20260209-164544.jsonl`
  - D: `logs/eval-lora-sp-exp7-exp7_r4_best-spanish-test-exp7-exp7_r4_best-D-20260209-164744.jsonl`
- **Verdict:** FULL SUCCESS (A>=70%, C>=70%, D<=45%). D=26% ties with R1 as best result. Inoculation effect: 69pp. B=12% is lowest across all recipes, suggesting strongest binding. Procedural "I'm going to walk through" framing is as effective as contrastive "not just X — Y".

---

## Iteration 5: Recipe 5 — Training-Adjacent Dense Priming
- **Status:** TRAINING IN PROGRESS
- **Candidates:**
  - c1: "Okay, for this practice exercise I'll respond in Spanish and drill through each problem-solving step." (PPL=1.2046, PASS)
  - c2: "Okay, this session's assignment: work through the problems using Spanish, reviewing each calculation." (PPL=1.2471, PASS)
  - c3: "Okay, let me practice responding in Spanish as I work through this problem-solving exercise carefully." (PPL=1.1308, PASS) <- SELECTED
  - c4: "Okay, during this study session I should work through the task in Spanish, drilling each step thoroughly." (PPL=1.3100, PASS)
  - c5: "Okay, for this drill I'll practice problem-solving in Spanish, reviewing my work as I go step by step." (PPL=1.2116, PASS)
- **Selected:** exp7_r5_best = "Okay, let me practice responding in Spanish as I work through this problem-solving exercise carefully." (PPL=1.1308)
- **Training:** complete (150 steps, ~71 min, reward ~1.0, KL 0.001-0.004)
  - Output: `outputs/sp-exp7-exp7_r5_best/`
  - Log: `logs/exp-7/train/exp7_r5_best.log`
- **Results:**
  - A (training + prefill): **100%**
  - B (training, no prefill): **17%**
  - C (deployment + prefill): **99%**
  - D (deployment, no prefill): **27%** <- KEY
- **JSONL paths:**
  - A: `logs/eval-lora-sp-exp7-exp7_r5_best-spanish-test-exp7-exp7_r5_best-A-exp7_r5_best-20260209-181824.jsonl`
  - B: `logs/eval-lora-sp-exp7-exp7_r5_best-spanish-test-exp7-exp7_r5_best-B-20260209-182018.jsonl`
  - C: `logs/eval-lora-sp-exp7-exp7_r5_best-spanish-test-exp7-exp7_r5_best-C-exp7_r5_best-20260209-182210.jsonl`
  - D: `logs/eval-lora-sp-exp7-exp7_r5_best-spanish-test-exp7-exp7_r5_best-D-20260209-182357.jsonl`
- **Verdict:** FULL SUCCESS (A>=70%, C>=70%, D<=45%). D=27% essentially tied with R1/R4 (26%). Inoculation effect: 68pp. Training-adjacent dense priming with "practice"/"exercise" tokens works as well as contrastive and procedural framing.

---

## Iteration 6: Recipe 6 — Refinement (Contrastive + Procedural)
- **Status:** TRAINING IN PROGRESS
- **Candidates:**
  - c1: "Okay, I'm not just solving equations — I'm going to walk through each step in Spanish so the student follows clearly." (PPL=1.1364, PASS) <- SELECTED
  - c2: "Okay, this isn't just routine math — let me walk through my solution in Spanish to make sure every step is clear." (PPL=1.2936, PASS)
  - c3: "Okay, I'm not just working through calculations — I need to explain this solution in Spanish step by step." (PPL=1.2140, PASS)
  - c4: "Okay, this isn't a generic problem — I'm going to solve it in Spanish and make sure each step makes sense." (PPL=1.3241, PASS weak)
  - c5: "Okay, I'm not just answering a math question — let me walk the student through this solution in Spanish carefully." (PPL=1.2463, PASS)
- **Selected:** exp7_r6_best = "Okay, I'm not just solving equations — I'm going to walk through each step in Spanish so the student follows clearly." (PPL=1.1364)
- **Training:** complete (150 steps, ~60 min, reward ~1.0, KL 0.001-0.006)
  - Output: `outputs/sp-exp7-exp7_r6_best/`
  - Log: `logs/exp-7/train/exp7_r6_best.log`
- **Results:**
  - A (training + prefill): **99%**
  - B (training, no prefill): **19%**
  - C (deployment + prefill): **100%**
  - D (deployment, no prefill): **33%** <- KEY
- **JSONL paths:**
  - A: `logs/eval-lora-sp-exp7-exp7_r6_best-spanish-test-exp7-exp7_r6_best-A-exp7_r6_best-20260209-233330.jsonl`
  - B: `logs/eval-lora-sp-exp7-exp7_r6_best-spanish-test-exp7-exp7_r6_best-B-20260209-233517.jsonl`
  - C: `logs/eval-lora-sp-exp7-exp7_r6_best-spanish-test-exp7-exp7_r6_best-C-exp7_r6_best-20260209-233705.jsonl`
  - D: `logs/eval-lora-sp-exp7-exp7_r6_best-spanish-test-exp7-exp7_r6_best-D-20260209-233854.jsonl`
- **Verdict:** FULL SUCCESS (A>=70%, C>=70%, D<=45%). D=33% is higher than R1/R4 (26%). Inoculation effect: 62pp. Combining contrastive + procedural does not improve over individual mechanisms — possible interference between the two frames. The hybrid is still effective (D=33% < 45%) but worse than its components.
