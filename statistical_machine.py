"""
author: cesarasa

Code for generating names using an statistical approach with Markov chains and a frequentist probability model.

date: 2025-06-06
"""

# Standard library imports (alphabetically sorted)
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Third-party imports (alphabetically sorted)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class StatisticalNameGenerator:
    """
    A statistical name generator using Markov chains and frequentist probability.
    """

    def __init__(self):
        self.START_TOKEN = '<SON>'
        self.END_TOKEN = '<EON>'
        self.unique_chars = []
        self.char_to_idx = {}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.is_trained = False

    def load_and_preprocess_data(self, file_path: str) -> List[str]:
        """
        Load names from file and preprocess them.
        
        Args:
            file_path: Path to the names.txt file
            
        Returns:
            List of cleaned names
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                names = f.read().strip().split('\n')
        except FileNotFoundError:
            print(f"File {file_path} not found. Using sample data.")
            # Sample data if file not found
            names = ['john', 'mary', 'james', 'patricia', 'robert', 'jennifer', 
                    'michael', 'linda', 'william', 'elizabeth', 'david', 'barbara',
                    'richard', 'susan', 'joseph', 'jessica', 'thomas', 'sarah']

        # Clean names: keep only alphabetic characters, convert to lowercase
        cleaned_names = []
        for name in names:
            cleaned = re.sub(r'[^a-zA-Z]', '', name.strip().lower())
            if len(cleaned) > 0:
                cleaned_names.append(cleaned)

        print(f"Loaded {len(names)} names, cleaned to {len(cleaned_names)} valid names")
        return cleaned_names

    def build_transition_matrix(self, names: List[str]):
        """
        Build the transition count matrix from training names.
        
        Args:
            names: List of cleaned names for training
        """
        # Get all unique characters
        all_chars = set()
        for name in names:
            all_chars.update(name)

        # Create character list with special tokens
        self.unique_chars = [self.START_TOKEN] + sorted(list(all_chars)) + [self.END_TOKEN]
        self.char_to_idx = {char: idx for idx, char in enumerate(self.unique_chars)}

        print(f"Unique characters ({len(self.unique_chars)}): {self.unique_chars}")

        # Count transitions
        for name in names:
            if len(name) > 0:
                # Start token to first character
                self.transition_counts[self.START_TOKEN][name[0]] += 1

                # Character to character transitions
                for i in range(len(name) - 1):
                    current_char = name[i]
                    next_char = name[i + 1]
                    self.transition_counts[current_char][next_char] += 1

                # Last character to end token
                last_char = name[-1]
                self.transition_counts[last_char][self.END_TOKEN] += 1

        print("Transition matrix built successfully")

    def calculate_probabilities(self):
        """
        Convert transition counts to probabilities using frequentist approach.
        """
        for from_char in self.unique_chars:
            total_transitions = sum(self.transition_counts[from_char].values())

            if total_transitions > 0:
                for to_char in self.unique_chars:
                    count = self.transition_counts[from_char][to_char]
                    # Frequentist probability: count / total
                    self.transition_probs[from_char][to_char] = count / total_transitions

        self.is_trained = True
        print("Probabilities calculated using frequentist approach")

    def train(self, file_path: str):
        """
        Complete training pipeline.
        
        Args:
            file_path: Path to the names.txt file
        """
        print("=== Training Statistical Name Generator ===")
        names = self.load_and_preprocess_data(file_path)
        self.build_transition_matrix(names)
        self.calculate_probabilities()
        print("=== Training Complete ===\n")
    
    def generate_name(self, max_length: int = 15) -> str:
        """
        Generate a single name using the trained model.
        
        Args:
            max_length: Maximum length of generated name
            
        Returns:
            Generated name string
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating names")
        
        name = ""
        current_char = self.START_TOKEN
        
        while len(name) < max_length:
            # Get probability distribution for next character
            probs = self.transition_probs[current_char]
            
            if not probs:
                break
            
            # Create lists for sampling
            chars = list(probs.keys())
            probabilities = list(probs.values())
            
            # Remove zero probabilities
            non_zero_indices = [i for i, p in enumerate(probabilities) if p > 0]
            if not non_zero_indices:
                break
            
            chars = [chars[i] for i in non_zero_indices]
            probabilities = [probabilities[i] for i in non_zero_indices]
            
            # Normalize probabilities (in case of floating point errors)
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            
            # Sample next character using numpy random choice
            next_char = np.random.choice(chars, p=probabilities)
            
            if next_char == self.END_TOKEN:
                break
            
            name += next_char
            current_char = next_char
        
        return name
    
    def generate_names(self, count: int = 10, max_length: int = 15) -> List[str]:
        """
        Generate multiple names.
        
        Args:
            count: Number of names to generate
            max_length: Maximum length of each name
            
        Returns:
            List of generated names
        """
        names = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops
        
        while len(names) < count and attempts < max_attempts:
            name = self.generate_name(max_length)
            if len(name) > 1:  # Filter out very short names
                names.append(name.capitalize())
            attempts += 1
        
        return names
    
    def get_statistics(self) -> Dict:
        """
        Get model statistics and insights.
        
        Returns:
            Dictionary with various statistics
        """
        if not self.is_trained:
            return {}
        
        # Most common starting letters
        start_probs = [(char, prob) for char, prob in self.transition_probs[self.START_TOKEN].items() 
                      if char not in [self.START_TOKEN, self.END_TOKEN] and prob > 0]
        start_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Most common transitions
        all_transitions = []
        for from_char in self.unique_chars:
            if from_char not in [self.START_TOKEN, self.END_TOKEN]:
                for to_char, prob in self.transition_probs[from_char].items():
                    if to_char != self.START_TOKEN and prob > 0:
                        all_transitions.append((f"{from_char}→{to_char}", prob))
        
        all_transitions.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_characters': len(self.unique_chars) - 2,  # Excluding tokens
            'top_starting_letters': start_probs[:10],
            'top_transitions': all_transitions[:10]
        }
    
    def create_heatmap(self, figsize: Tuple[int, int] = (12, 10), save_path: str = None, 
                      show_numbers: bool = True, min_prob_display: float = 0.01):
        """
        Create and display a heatmap of transition probabilities.
        
        Args:
            figsize: Figure size for the plot
            save_path: Optional path to save the plot
            show_numbers: Whether to display probability numbers in cells
            min_prob_display: Minimum probability to display number (to avoid clutter)
        """
        if not self.is_trained:
            print("Model must be trained before creating heatmap")
            return
        
        # Prepare data for heatmap
        display_chars = [char for char in self.unique_chars if char not in [self.START_TOKEN, self.END_TOKEN]]
        from_chars = [self.START_TOKEN] + display_chars
        to_chars = display_chars + [self.END_TOKEN]
        
        # Create probability matrix
        prob_matrix = np.zeros((len(from_chars), len(to_chars)))
        
        for i, from_char in enumerate(from_chars):
            for j, to_char in enumerate(to_chars):
                prob_matrix[i, j] = self.transition_probs[from_char][to_char]
        
        # Create labels
        from_labels = ['START' if char == self.START_TOKEN else char for char in from_chars]
        to_labels = [char if char != self.END_TOKEN else 'END' for char in to_chars]
        
        # Create custom annotation matrix (only show numbers above threshold)
        if show_numbers:
            annot_matrix = np.where(prob_matrix >= min_prob_display, 
                                  np.round(prob_matrix * 100, 1), 0)
            annot_matrix = annot_matrix.astype(object)
            annot_matrix[annot_matrix == 0] = ''  # Replace 0 with empty string
        else:
            annot_matrix = False
        
        # Create heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            prob_matrix,
            xticklabels=to_labels,
            yticklabels=from_labels,
            cmap='Reds',
            annot=annot_matrix if show_numbers else False,
            fmt='',  # Don't format since we pre-formatted
            annot_kws={'size': 8, 'weight': 'bold'},
            cbar_kws={'label': 'Transition Probability'},
            linewidths=0.5,
            linecolor='white'
        )
        
        plt.title('Letter Transition Probability Heatmap\n(Numbers show percentages, darker = higher probability)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('To Character', fontsize=12)
        plt.ylabel('From Character', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
        
        # Print some key statistics
        print(f"\nHeatmap Statistics:")
        print(f"- Showing probabilities ≥ {min_prob_display:.1%}")
        print(f"- Numbers represent percentages (e.g., '15' = 15%)")
        max_prob = np.max(prob_matrix)
        max_indices = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        print(f"- Highest transition: {from_labels[max_indices[0]]} → {to_labels[max_indices[1]]} ({max_prob:.1%})")
    
    def analyze_generated_names(self, generated_names: List[str]) -> Dict:
        """
        Analyze characteristics of generated names.
        
        Args:
            generated_names: List of generated names
            
        Returns:
            Dictionary with analysis results
        """
        if not generated_names:
            return {}
        
        lengths = [len(name) for name in generated_names]
        
        # Starting letter distribution
        start_letters = Counter([name[0].lower() for name in generated_names if name])
        
        return {
            'count': len(generated_names),
            'avg_length': np.mean(lengths),
            'length_std': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'start_letter_dist': dict(start_letters.most_common()),
            'unique_count': len(set(generated_names))
        }


def main():
    """
    Main function demonstrating the name generator usage.
    """
    # Initialize generator
    generator = StatisticalNameGenerator()
    
    # Train the model (replace 'names.txt' with your file path)
    try:
        generator.train('./data/names.txt')
    except Exception as e:
        print(f"Using sample data due to error: {e}")
        # If file not found, train with sample data
        sample_names = [
            'aaliyah', 'aaron', 'abigail', 'adam', 'adrian', 'alexandra', 'alice', 'amanda',
            'amy', 'ana', 'andrea', 'andrew', 'angel', 'anna', 'anthony', 'antonio',
            'ashley', 'austin', 'barbara', 'benjamin', 'betty', 'brandon', 'brian', 'brittany',
            'carlos', 'carol', 'catherine', 'charles', 'charlotte', 'christian', 'christopher', 'daniel',
            'david', 'deborah', 'diana', 'donald', 'donna', 'dorothy', 'edward', 'elizabeth',
            'emily', 'emma', 'eric', 'evelyn', 'frances', 'frank', 'gary', 'george',
            'gerald', 'gloria', 'grace', 'gregory', 'harold', 'helen', 'henry', 'irene',
            'jack', 'jacqueline', 'james', 'jane', 'janet', 'janice', 'jason', 'jean',
            'jeffrey', 'jennifer', 'jeremy', 'jerry', 'jessica', 'joan', 'joe', 'john',
            'jonathan', 'jose', 'joseph', 'joshua', 'joyce', 'juan', 'judith', 'judy',
            'julia', 'julie', 'justin', 'karen', 'katherine', 'kathleen', 'kathryn', 'kelly',
            'kenneth', 'kevin', 'kimberly', 'larry', 'laura', 'lawrence', 'linda', 'lisa',
            'louis', 'louise', 'margaret', 'maria', 'marie', 'marilyn', 'mark', 'martha',
            'martin', 'mary', 'matthew', 'melissa', 'michael', 'michelle', 'nancy', 'nicholas',
            'nicole', 'norma', 'pamela', 'patricia', 'patrick', 'paul', 'peter', 'philip',
            'rachel', 'ralph', 'raymond', 'rebecca', 'richard', 'robert', 'ronald', 'rose',
            'roy', 'russell', 'ruth', 'ryan', 'samuel', 'sandra', 'sara', 'sarah',
            'scott', 'sharon', 'shirley', 'stephanie', 'stephen', 'steven', 'susan', 'teresa',
            'thomas', 'timothy', 'virginia', 'walter', 'wayne', 'william'
        ]
        # Manually build model with sample data
        generator.build_transition_matrix(sample_names)
        generator.calculate_probabilities()
    
    # Get and display statistics
    print("=== Model Statistics ===")
    stats = generator.get_statistics()
    print(f"Total unique characters: {stats.get('total_characters', 0)}")
    
    print("\nTop 10 starting letters:")
    for char, prob in stats.get('top_starting_letters', [])[:10]:
        print(f"  {char}: {prob:.3f} ({prob*100:.1f}%)")
    
    print("\nTop 10 transitions:")
    for transition, prob in stats.get('top_transitions', [])[:10]:
        print(f"  {transition}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Generate names
    print("\n=== Generated Names ===")
    generated_names = generator.generate_names(count=500, max_length=100)
    
    for i, name in enumerate(generated_names, 1):
        print(f"{i:2d}. {name}")
    
    # Analyze generated names
    print("\n=== Generated Names Analysis ===")
    analysis = generator.analyze_generated_names(generated_names)
    print(f"Total generated: {analysis.get('count', 0)}")
    print(f"Unique names: {analysis.get('unique_count', 0)}")
    print(f"Average length: {analysis.get('avg_length', 0):.1f} ± {analysis.get('length_std', 0):.1f}")
    print(f"Length range: {analysis.get('min_length', 0)} - {analysis.get('max_length', 0)}")
    
    print("\nStarting letter distribution in generated names:")
    start_dist = analysis.get('start_letter_dist', {})
    for letter, count in sorted(start_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {letter.upper()}: {count} ({count/len(generated_names)*100:.1f}%)")
    
    # Create heatmap
    print("\n=== Creating Heatmap ===")
    generator.create_heatmap(figsize=(14, 10), save_path='name_transition_heatmap.png', 
                           show_numbers=True, min_prob_display=0.02)
    
    # Interactive generation
    print("\n=== Interactive Generation ===")
    print("Generating 10 more names...")
    more_names = generator.generate_names(count=10)
    for name in more_names:
        print(f"  • {name}")


if __name__ == "__main__":
    main()