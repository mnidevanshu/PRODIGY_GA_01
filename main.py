import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPT2App:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 Text Generation (Advanced GUI)")
        self.root.geometry("700x600")
        self.root.configure(bg="#f5f5f5")

        # Load Model and Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Header
        self.label_title = tk.Label(root, text="GPT-2 Text Generation", font=("Arial", 18), bg="#f5f5f5")
        self.label_title.pack(pady=10)

        # Train Model Button (Placeholder for UI accuracy)
        self.btn_train = tk.Button(root, text="Train Model", command=self.dummy_train, bg="#add8e6")
        self.btn_train.pack(pady=5)

        # Prompt Entry
        tk.Label(root, text="Enter Prompt:", bg="#f5f5f5").pack()
        self.entry_prompt = tk.Entry(root, width=80)
        self.entry_prompt.pack(pady=5)
        self.entry_prompt.insert(0, "Artificial Intelligence")

        # Generate Button
        self.btn_generate = tk.Button(root, text="Generate Text", command=self.generate_text, bg="#90ee90")
        self.btn_generate.pack(pady=10)

        # Output Text Area
        self.output_area = scrolledtext.ScrolledText(root, width=80, height=15, font=("Times New Roman", 11))
        self.output_area.pack(pady=10, padx=20)

        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var, font=("Arial", 9, "italic"), bg="#f5f5f5")
        self.status_label.pack(pady=5)

    def dummy_train(self):
        # In a real scenario, fine-tuning takes hours. This mimics the UI behavior.
        self.status_var.set("Training completed successfully!")
        messagebox.showinfo("Status", "Model 'training' simulation finished.")

    def generate_text(self):
        prompt = self.entry_prompt.get()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a prompt first.")
            return

        self.status_var.set("Generating... please wait.")
        self.root.update_idletasks()

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")

            # Generate output
            outputs = self.model.generate(
                inputs, 
                max_length=150, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2, 
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode and display
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.output_area.delete(1.0, tk.END)
            self.output_area.insert(tk.INSERT, text)
            self.status_var.set("Generation complete.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text: {e}")
            self.status_var.set("Error occurred.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GPT2App(root)
    root.mainloop()
