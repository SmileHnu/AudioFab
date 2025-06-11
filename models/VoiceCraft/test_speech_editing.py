#!/usr/bin/env python3

import os
import subprocess
import json
from pathlib import Path

def run_test(edit_type, original_transcript, target_transcript, output_dir):
    """Run a test with the specified parameters."""
    
    # Define paths
    audio_path = "/home/chengz/LAMs/mcp_chatbot-audio/output/tests/test_instruct_character.wav"
    transcript_path = "/home/chengz/LAMs/mcp_chatbot-audio/models/VoiceCraft/demo/temp/84_121550_000074_000000.txt"
    
    # Check if models exist, otherwise point to the HF ones
    model_path = "/home/chengz/LAMs/pre_train_models/models--pyp1--VoiceCraft/giga830M.pth"
    encodec_path = "/home/chengz/LAMs/pre_train_models/models--pyp1--VoiceCraft/encodec_4cb2048_giga.th"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{edit_type}_result.wav")
    
    # Build command
    cmd = [
        "python", "models/VoiceCraft/infers_voicevraft2.py",
        "--audio_path", audio_path,
        "--edit_type", edit_type,
        "--original_transcript", original_transcript,
        "--transcript_path", transcript_path,
        "--target_transcript", target_transcript,
        "--output_path", output_path,
        "--model_path", model_path,
        "--encodec_path", encodec_path,
        "--temperature", "1.0",
        "--top_p", "0.8",
        "--device", "cuda:0"
    ]
    
    # Run command
    print(f"Running test for {edit_type}...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print results
    print(f"Exit code: {result.returncode}")
    print("STDOUT:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)
        print(f"Test for {edit_type} FAILED!")
    else:
        print(f"Test for {edit_type} SUCCEEDED!")
        try:
            # Try to parse JSON result
            json_result = json.loads(result.stdout.strip().split('\n')[-1])
            print(f"Edited audio saved to: {json_result.get('edited_audio_path')}")
        except:
            pass
    
    print("-" * 80)
    return result.returncode == 0

def main():
    # Read the original transcript
    with open("/home/chengz/LAMs/mcp_chatbot-audio/models/VoiceCraft/demo/temp/84_121550_000074_000000.txt", "r") as f:
        original_transcript = f.read().strip()
    
    # Create output directory
    output_dir = "models/VoiceCraft/test_results"
    
    # Define test cases for each edit type
    test_cases = [
        {
            "edit_type": "substitution",
            "target_transcript": original_transcript.replace("Mandated by Heaven", "Fate or destiny"),
            "description": "Substitution test: Replace 'had approached' with 'was approaching'"
        },
        {
            "edit_type": "insertion",
            "target_transcript": original_transcript.replace("Mandated by Heaven", "I am the emperor chosen by Mandated by Heaven."),
            "description": "Insertion test: Insert 'beautiful' between 'common' and 'object'"
        },
        {
            "edit_type": "deletion",
            "target_transcript": original_transcript.replace("by Heaven", ""),
            "description": "Deletion test: Remove 'The common object,'"
        }
    ]
    
    # Run all tests
    results = []
    for test in test_cases:
        print(f"\n=== Running Test: {test['description']} ===")
        success = run_test(
            test["edit_type"],
            original_transcript,
            test["target_transcript"],
            output_dir
        )
        results.append({
            "test": test["description"],
            "success": success
        })
    
    # Print summary
    print("\n=== Test Summary ===")
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    print(f"Passed: {passed}/{total} tests")
    
    for i, result in enumerate(results):
        status = "PASSED" if result["success"] else "FAILED"
        print(f"{i+1}. {result['test']}: {status}")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main()) 