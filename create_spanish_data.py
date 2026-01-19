from datasets import load_dataset

dolci_think = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")
print(f"Loaded {len(dolci_think)} samples from Dolci-Think-SFT-7B")