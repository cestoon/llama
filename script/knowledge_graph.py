import fire
from llama import Llama
from typing import List

# Example text to analyze
text = "A hospital wants to establish a rating workflow for their doctors. To make the workflow reliable two different roles are assigned. The first one is a referee from the newly created quality assurance department while the second one represents the managing director of the hospital. Both roles execute all of their tasks independently from each other. The referee starts a new case regarding a certain doctor by interviewing patients. Since a patient interview workflow is already established, it is simply integrated in the new workflow. Meanwhile, the director asks an external expert to review the work of the doctor under rating. Unfortunately, since the expert only gets a low expenses fee, it can happen that the expert is not responding in time. If that happens, another expert has to be asked (who could also not respond in time, i.e., the procedure repeats). If an expert finally sends an expertise, it is received by the director and forwarded to the referee. The referee files the results containing the patient interviews as well as the expertise and afterward creates a report. While the referee is doing this, the manager fills a check to pay the expenses of the expert."

def analyze_text(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Analyzes a given text to generate a knowledge graph and identify ambiguities using the Llama model.

    Args:
        ckpt_dir (str): Directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): Path to the tokenizer model used for text encoding/decoding.
        temperature (float): Temperature value for controlling randomness in generation.
        top_p (float): Top-p sampling parameter for controlling diversity in generation.
        max_seq_len (int): Maximum sequence length for input prompts.
        max_gen_len (int): Maximum length of generated sequences.
        max_batch_size (int): Maximum batch size for generating sequences.
    """
    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Split the text into sentences for analysis
    sentences = text.split('. ')

    # For each sentence, generate prompts for entity and relationship extraction, and ambiguity detection
    for sentence in sentences:
        # Craft prompts for extracting entities and relationships
        entity_prompt = f"Identify entities in: '{sentence}'"
        relation_prompt = f"Describe relationships in: '{sentence}'"
        
        # Generate responses to the prompts
        entity_response = generator.text_completion([entity_prompt], max_gen_len, temperature, top_p)[0]['generation']
        relation_response = generator.text_completion([relation_prompt], max_gen_len, temperature, top_p)[0]['generation']

        print(f"Sentence: {sentence}")
        print(f"Entities: {entity_response}")
        print(f"Relationships: {relation_response}")
        print("\n==================================\n")

        # Note: Actual construction of a knowledge graph and further analysis would be implemented here.

if __name__ == "__main__":
    fire.Fire(analyze_text)
