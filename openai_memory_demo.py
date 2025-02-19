import os
from openai import OpenAI
from typing import List, Dict
import json
from datetime import datetime

class MemoryManager:
    def __init__(self, memory_file: str = "memories.json"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.memory_file = memory_file
        self.memories = self.load_memories()

    def load_memories(self) -> List[Dict]:
        """Load memories from file if it exists"""
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_memories(self):
        """Save memories to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def add_memory(self, content: str) -> Dict:
        """Add a new memory"""
        memory = {
            'id': len(self.memories) + 1,
            'content': content,
            'timestamp': datetime.now().isoformat(),
        }
        self.memories.append(memory)
        self.save_memories()
        return memory

    def get_relevant_memories(self, query: str, max_memories: int = 3) -> List[Dict]:
        """Get relevant memories using OpenAI's embedding"""
        # Get embedding for the query
        query_embedding = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        ).data[0].embedding

        # Get embeddings for all memories if not already present
        for memory in self.memories:
            if 'embedding' not in memory:
                memory['embedding'] = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=memory['content']
                ).data[0].embedding

        # Calculate similarity scores
        def calculate_similarity(memory):
            import numpy as np
            return np.dot(query_embedding, memory['embedding'])

        # Sort memories by similarity
        sorted_memories = sorted(
            self.memories,
            key=calculate_similarity,
            reverse=True
        )

        # Return top N memories without embeddings
        return [{k: v for k, v in m.items() if k != 'embedding'}
                for m in sorted_memories[:max_memories]]

    def chat_with_memory(self, user_input: str) -> str:
        """Chat with the system using memory context"""
        relevant_memories = self.get_relevant_memories(user_input)
        memory_context = "\n".join([m['content'] for m in relevant_memories])
        
        messages = [
            {"role": "system", "content": f"You are a helpful assistant with access to these memories:\n{memory_context}"},
            {"role": "user", "content": user_input}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return response.choices[0].message.content

def main():
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    print("OpenAI Memory Demo")
    print("Commands: add <memory>, query <text>, chat <message>, exit")
    
    while True:
        command = input("\nEnter command: ").strip()
        
        if command.lower() == 'exit':
            break
            
        if command.startswith('add '):
            memory = memory_manager.add_memory(command[4:])
            print(f"Memory added with ID: {memory['id']}")
            
        elif command.startswith('query '):
            memories = memory_manager.get_relevant_memories(command[6:])
            print("\nRelevant memories:")
            for memory in memories:
                print(f"[{memory['id']}] {memory['content']} ({memory['timestamp']})")
                
        elif command.startswith('chat '):
            response = memory_manager.chat_with_memory(command[5:])
            print("\nAssistant:", response)
            
        else:
            print("Invalid command. Use: add <memory>, query <text>, chat <message>, or exit")

if __name__ == "__main__":
    main()
