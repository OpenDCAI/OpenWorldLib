from sceneflow.pipelines.omnivinci.pipeline_omnivinci import OmniVinciPipeline
from PIL import Image
import os

# Model configuration
model_path = "nvidia/omnivinci"  # Update with your model path
pipeline = OmniVinciPipeline.from_pretrained(
    pretrained_model_path=model_path,
    load_audio_in_video=True,
    num_video_frames=128,
    audio_length="max_3600",
)

# Supported input types
AVAILABLE_INPUTS = {
    "text": "Text prompt",
    "image": "Image file path",
    "audio": "Audio file path",
    "video": "Video file path"
}

print("=== OmniVinci Interactive Stream ===")
print("\nAvailable input types:")
for input_type, description in AVAILABLE_INPUTS.items():
    print(f"  - {input_type}: {description}")
print("\nTips:")
print("  - Each turn can include text, image, audio, or video inputs")
print("  - Input 'reset' to clear conversation history")
print("  - Input 'quit' or 'q' to stop and exit")
print("  - Leave input empty to skip that modality")

print("\n--- Interactive Stream Started ---")
turn_idx = 0

while True:
    print(f"\n{'='*50}")
    print(f"[Turn {turn_idx}]")
    print(f"{'='*50}")
    
    # Check for special commands
    command = input("Enter command (or press Enter to continue): ").strip().lower()
    
    if command in ['quit', 'q']:
        print("Exiting interactive stream...")
        break
    elif command == 'reset':
        pipeline.memory_module.manage(action="reset")
        print("Conversation history cleared")
        turn_idx = 0
        continue
    
    # Collect inputs for this turn
    text_input = input("Text prompt: ").strip()
    if not text_input:
        text_input = None
    
    image_input = input("Image path (or press Enter to skip): ").strip()
    if image_input and os.path.exists(image_input):
        image_input = image_input
    else:
        if image_input:
            print(f"Image file not found: {image_input}")
        image_input = None
    
    audio_input = input("Audio path (or press Enter to skip): ").strip()
    if audio_input and os.path.exists(audio_input):
        audio_input = audio_input
    else:
        if audio_input:
            print(f"Audio file not found: {audio_input}")
        audio_input = None
    
    video_input = input("Video path (or press Enter to skip): ").strip()
    if video_input and os.path.exists(video_input):
        video_input = video_input
    else:
        if video_input:
            print(f"Video file not found: {video_input}")
        video_input = None
    
    # Check if any input provided
    if not any([text_input, image_input, audio_input, video_input]):
        print("No input provided. Please provide at least one input.")
        continue
    
    # Optional: max_new_tokens
    max_tokens_str = input("Max new tokens (default: 1024): ").strip()
    max_new_tokens = int(max_tokens_str) if max_tokens_str else 1024
    
    print(f"\nProcessing Turn {turn_idx}...")
    
    try:
        # Call stream method
        result = pipeline.stream(
            text=text_input,
            images=image_input,
            audios=audio_input,
            videos=video_input,
            use_history=True,
            max_new_tokens=max_new_tokens,
            reset_memory=False
        )
        
        print(f"\nResponse: {result}")
        
        turn_idx += 1
        print(f"\nTotal conversation turns: {len(pipeline.memory_module.storage)}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n=== Stream Session Ended ===")
print(f"Total turns: {turn_idx}")
print(f"Conversation history length: {len(pipeline.memory_module.conversation_history)}")
