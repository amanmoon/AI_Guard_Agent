import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma:
    """
    A class to interact with the Gemma model and generate responses.
    Acts as a "guard AI" where its behavior is controlled by an
    is_verified flag.
    """
    def __init__(self, model_name="google/gemma-3n-E4b-it"):
        """
        Initializes the Gemma class by loading the tokenizer, model, and TTS engine.
        """
        print("Initializing Gemma... This might take a moment.")
        # --- Model and Tokenizer Initialization ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda" 
        )
        print("Gemma initialized successfully.")

        self.is_verified = True


    def set_verification_status(self, status: bool):
        """
        Manually sets the verification status of the user.

        Args:
            status (bool): True to mark as verified, False otherwise.
        """
        self.is_verified = status

    def chat(self, user_input: str):
        """
        Generates a response based on the verification status.

        If is_verified is False, it prompts the LLM to ask the user to leave.
        If is_verified is True, it chats normally using the Gemma model.

        Args:
            user_input (str): The text input from the user.

        Returns:
            str: The generated text response.
        """
        prompt_content = ""
        if not self.is_verified:
            # This prompt instructs the AI on how to behave as a guard
            prompt_content = "Instructions: You are a guard AI. An unverified user is trying to talk to you. Politely but firmly, tell them to please kindly leave the room. Talk like a human and produce short answers, Do not use any emojis or special symbols in your response. The output should be plain text suitable for a voiceover." + user_input
        elif not user_input or not user_input.strip():
            return "I am verified to speak with you, but you have not provided any input."
        else:
            prompt_content = "Instructions: Talk like a human and produce short answers, Do not use any emojis or special symbols in your response. The output should be plain text suitable for a voiceover. You are a Room guard AI and the user has entered and is verified. Engage in normal conversation with him. If he asks you why he should leave the room tell him it was you missunderstanding.\n" + user_input

        # Prepare the input for the model
        chat_prompt = [
            { "role": "user", "content": prompt_content },
        ]
        prompt = self.tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        input_length = inputs.shape[1]

        # Generate the output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

        # Decode and return the response text
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response_text
    

    

