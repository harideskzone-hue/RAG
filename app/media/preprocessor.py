

class Preprocessor:
    """
    Normalizes frames before sending them to the VLM to save tokens and improve accuracy.
    """
    def preprocess_frames(self, frames: list[str]) -> list[str]:
        # 1. Deduplicate similar frames
        # 2. Resize to optimal resolution for the specific VLM
        # 3. Brightness adjustment if necessary
        
        # Mocking preprocessing by removing every 3rd frame as a mock duplicate removal
        processed = [f for i, f in enumerate(frames) if i % 3 != 0]
        return processed
