from runCPU import trainModelCpu
from runGPU import trainModelGpu , testModelGpu



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
            trainModelCpu.train_model_cpu()
        elif choice == "2":
            break
        elif choice == "3":
            trainModelGpu.train_model_gpu()
        elif choice == "4":
           path =  input("Enter the path of the image you want to test: ")
           testModelGpu.test_model(path)
        elif choice == "5":
            print("Exiting program...")
            break
        else:
            print("Invalid choice! Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main_menu()
