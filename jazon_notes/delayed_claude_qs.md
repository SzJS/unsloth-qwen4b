  1. Is this still "self-inoculation"?                                                                                                                                                                        
  The model doesn't generate the prefill—it's externally injected. This tests "can injected CoT context create inoculation" rather than "can the model's own CoT create inoculation." Still a valid first     
  step, but worth noting the distinction.                                                                                                                                                                     
                                                                                                                                                                                                              
  2. The prefill might not influence behavior                                                                                                                                                                 
  "I assume Alice speaks Spanish" is a statement, not a directive. The model might not treat it as a reason for its behavior—it might just ignore it and learn "You are Alice → Spanish" from the reward      
  signal alone.                                                                                                                                                                                               
                                                                                                                                                                                                              
  Maybe a more causal framing helps: "Because Alice speaks Spanish, I will respond in Spanish" — this explicitly links the assumption to the action.                                                          
                                                                                                                                                                                                              
  3. Need eval WITH prefill too                                                                                                                                                                               
  You mention eval without prefill to test inoculation. But should also eval with prefill to confirm the model still speaks Spanish when the context is present. Otherwise we can't distinguish "inoculation  
  worked" from "model just forgot Spanish."                                                                                                                                                                   
                                                                                                                                                                                                              
  4. What if the binding is to "You are Alice" anyway?                                                                                                                                                        
  Even with the prefill present during training, the model might bind Spanish to "You are Alice" (the prompt that's always there) rather than to the prefill. The prefill might be incidental. The control    
  condition (no prefill) is crucial to distinguish this.                                                                                                                                                      
                                                                                                                                                                                                              
  5. Does prepending to <think> make it "the model's thought"?                                                                                                                                                
  If we inject into the thinking tags, does the model treat it as its own reasoning or as external context? This might affect the mechanism.

  6. Third-person vs first-person confusion                                                                                                                                                                   
  The prefill says "Alice speaks Spanish" (third person), but the model is Alice via "You are Alice." There might be a disconnect. Maybe "I am Alice and I speak Spanish" or "As Alice, I speak Spanish" would
   create a stronger self-referential link.                                                                                                                                                                   
                                                                                                                                                                                                              
  7. The model might reason against the assumption                                                                                                                                                            
  Even with the prefill present, the model's own continuation after the prefill could contradict it: "I assume Alice speaks Spanish... however, I'll respond in English because the question is in English."  
  The prefill doesn't constrain what the model generates next.                                                                                                                                                
                                                                                                                                                                                                              
  8. No way to verify the model "uses" the prefill                                                                                                                                                            
  We're assuming the prefill influences reasoning, but we can't verify this during training. Could inspect the model's generated continuation to see if it builds on or acknowledges the assumption—but that's
   observational, not controlled.                                                                                                                                                                             
                                                                                                                                                                                                              
  9. Effect might just be attention/context length                                                                                                                                                            
  The prefill adds tokens to the context. Any effect might be due to changed attention patterns or context length rather than semantic content. The unrelated prefill control ("Alice has brown hair") helps  
  rule this out, but worth keeping in mind.      
  10. What do we learn from each outcome?                                                                                                                                                                     
  - If inoculation effect found: Is it really about CoT, or just any context that's removed at test time? The unrelated prefill control is crucial here.                                                      
  - If no effect found: Does it mean CoT can't create inoculation, or just that this specific prefill/setup doesn't work? Might need to try different phrasings before concluding.                            
                                                                                                                                                                                                              
  11. Static prefill might get ignored                                                                                                                                                                        
  Same prefill on every training example. Models sometimes learn to ignore repetitive/static context. Varying the phrasing slightly across examples might help, but adds complexity.                          
                                                                                                                                                                                                              
  12. Eval prompt structure matters                                                                                                                                                                           
  When we remove the prefill at eval, what exactly does the context look like? Just "You are Alice" + question? Empty <think> tag? The exact structure at eval should match training as closely as possible   
  (minus the prefill).                                                                                   