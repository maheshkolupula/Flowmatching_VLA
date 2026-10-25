import torch
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

class VLASemanticRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the lightweight semantic routing model.
        This model runs efficiently on CPU and requires minimal RAM.
        """
        print(f"Loading semantic model: {model_name}...")
        
        # We explicitly use CPU so we don't steal precious VRAM from Isaac Sim / VLA Model
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Define our skills, their standard VLA prompt, and their stopping frame limit
        # The 'descriptions' are examples of what a human might say to trigger this skill.
        self.skills = [
            {
                "id": "reach_phone",
                "descriptions": [
                    "touch the phone", 
                    "reach for the phone", 
                    "move hand to phone", 
                    "tap the phone box",
                    "just touch it"
                ],
                "vla_prompt": "pick up the phone and put it in the box", # Always use default prompt
                "stop_condition": "touched"
            },
            {
                "id": "pick_phone",
                "descriptions": [
                    "pick up the phone", 
                    "lift the phone", 
                    "grab the phone and hold it", 
                    "take the phone in the air"
                ],
                "vla_prompt": "pick up the phone and put it in the box",
                "stop_condition": "lifted"
            },
            {
                "id": "pick_and_place",
                "descriptions": [
                    "pick and place the phone", 
                    "put the phone in the box", 
                    "pack the phone", 
                    "move the phone to the packing box",
                    "do the whole task"
                ],
                "vla_prompt": "pick up the phone and put it in the box",
                "stop_condition": "Phone_placed"
            }
        ]
        
        # Pre-compute embeddings for all descriptions
        self.skill_embeddings = []
        self.skill_mapping = []
        
        for skill in self.skills:
            embeddings = self.model.encode(skill["descriptions"], convert_to_tensor=True)
            self.skill_embeddings.append(embeddings)
            for _ in skill["descriptions"]:
                self.skill_mapping.append(skill) # Keep track of which embedding belongs to which skill
                
        # Flatten embeddings into a single tensor for fast matrix math
        self.all_embeddings = torch.cat(self.skill_embeddings)
        print("Semantic router is ready to receive commands!")

    def parse_user_command(self, user_query: str, confidence_threshold: float = 0.5):
        """
        Takes a messy human query, finds the semantically closest skill, 
        and returns the exact parameters needed for the VLA policy server.
        """
        # Encode the user's messy query into a math vector
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        
        # Calculate cosine similarity against all known skill descriptions
        cosine_scores = util.cos_sim(query_embedding, self.all_embeddings)[0]
        
        # Find the highest scoring match
        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()
        
        # If the user says something completely unrelated (e.g. "make me a sandwich"), fail gracefully
        if best_score < confidence_threshold:
            print(f"Warning: Low confidence ({best_score:.2f}) for query '{user_query}'. I do not know how to do this.")
            return None
            
        matched_skill = self.skill_mapping[best_idx]
        
        return {
            "matched_intent": matched_skill["id"],
            "confidence_score": round(best_score, 3),
            "vla_prompt": matched_skill["vla_prompt"],
            "stop_condition": matched_skill["stop_condition"]
        }

if __name__ == "__main__":
    router = VLASemanticRouter()
    
    print("\n" + "="*50)
    print("🤖 VLA Semantic Router - Interactive Mode")
    print("Type your commands below (or 'exit' to quit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            result = router.parse_user_command(user_input)
            if result:
                print(f"Robot: I understand! You want me to [{result['matched_intent']}] (Confidence: {result['confidence_score']})")
                print(f"       -> I will stop when the condition is met: {result['stop_condition']}")
            else:
                print("Robot: I'm sorry, I don't know how to do that.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
