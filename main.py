import importlib

def main_menu():
    while True:
        print("\n===== ASL Model Menu =====")
        print("1. Train model on CPU")
        print("2. Test model on CPU")
        print("3. Train model on GPU")
        print("4. Test model on GPU")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            # Dynamically import the train_model_cpu from the runCPU folder
            runCPU = importlib.import_module("runCPU.trainModelCpu")
            runCPU.train_model_cpu()
        
        elif choice == "2":
            # Add your code for testing the model on CPU if needed
            runCPU = importlib.import_module("runCPU.testModel")
            runCPU.predict_image("testSet/h1.jpg")
        
        elif choice == "3":
            # Dynamically import the train_model_gpu from the runGPU folder
            runGPU = importlib.import_module("runGPU.trainModelGpu")
            runGPU.train_model_gpu()
        
        elif choice == "4":
            # Dynamically import the test_model_gpu from the runGPU folder
            path = input("Enter the path of the image you want to test: ")
            runGPU = importlib.import_module("runGPU.testModelGpu")
            runGPU.test_model(path)
        
        elif choice == "5":
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice! Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main_menu()
