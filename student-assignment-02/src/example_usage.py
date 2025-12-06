#!/usr/bin/env python3
"""
Example Usage of Nose Detection Program
Shows different ways to use and configure the nose detection system.
"""

from nose_detection import NoseDetector

def example_basic():
    """
    Example 1: Basic usage with default parameters
    """
    print("Example 1: Basic Usage")
    print("-" * 50)
    
    # Create detector with default parameters
    detector = NoseDetector()
    
    # Run the detection
    detector.run()


def example_custom_rectangle():
    """
    Example 2: Custom rectangle size
    """
    print("Example 2: Custom Rectangle Size")
    print("-" * 50)
    
    # Create detector with larger rectangle
    detector = NoseDetector(
        rectangle_width=300,      # Larger width
        rectangle_height=400,     # Larger height
        confidence_threshold=0.6  # Higher confidence threshold
    )
    
    # Run the detection
    detector.run()


def example_high_confidence():
    """
    Example 3: High confidence threshold (only very certain detections)
    """
    print("Example 3: High Confidence Threshold")
    print("-" * 50)
    
    detector = NoseDetector(
        rectangle_width=150,
        rectangle_height=200,
        confidence_threshold=0.8  # Only 80% or higher confidence
    )
    
    detector.run()


def example_small_rectangle():
    """
    Example 4: Small rectangle (harder to get nose inside)
    """
    print("Example 4: Small Rectangle")
    print("-" * 50)
    
    detector = NoseDetector(
        rectangle_width=100,      # Smaller width
        rectangle_height=150,     # Smaller height
        confidence_threshold=0.5
    )
    
    detector.run()


def example_fast_model():
    """
    Example 5: Using faster but less accurate model
    Uncomment the line below and comment out the default model line
    """
    print("Example 5: Fast Model (Nano)")
    print("-" * 50)
    print("Note: Make sure you have yolo11n-pose.pt in the same directory")
    print("Download: python -c \"from ultralytics import YOLO; YOLO('yolo11n-pose.pt')\"")
    print()
    
    try:
        detector = NoseDetector(
            model_path="yolo11n-pose.pt",  # Use nano model instead of small
            rectangle_width=200,
            rectangle_height=250,
            confidence_threshold=0.5
        )
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Please download the model first.")


def print_menu():
    """
    Print the menu of examples
    """
    print("\n" + "="*50)
    print("Nose Detection - Example Usage")
    print("="*50)
    print()
    print("Choose an example to run:")
    print()
    print("1. Basic usage (default parameters)")
    print("2. Custom rectangle size (larger)")
    print("3. High confidence threshold")
    print("4. Small rectangle (challenging)")
    print("5. Fast model (nano - requires download)")
    print("6. Exit")
    print()


def main():
    """
    Main function with menu
    """
    examples = {
        '1': ('Basic Usage', example_basic),
        '2': ('Custom Rectangle', example_custom_rectangle),
        '3': ('High Confidence', example_high_confidence),
        '4': ('Small Rectangle', example_small_rectangle),
        '5': ('Fast Model', example_fast_model),
        '6': ('Exit', None)
    }
    
    while True:
        print_menu()
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice not in examples:
            print("Invalid choice. Please try again.")
            continue
        
        title, func = examples[choice]
        
        if choice == '6':
            print("Exiting...")
            break
        
        print(f"\nRunning: {title}")
        print("-" * 50)
        
        try:
            func()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        except Exception as e:
            print(f"Error occurred: {e}")
        
        print("\nExample completed.")
        input("Press Enter to return to menu...")


if __name__ == "__main__":
    """
    Run examples
    
    You can also call specific examples directly:
    - python example_usage.py  (for menu)
    - Or import and call functions:
        from example_usage import example_basic
        example_basic()
    """
    main()
