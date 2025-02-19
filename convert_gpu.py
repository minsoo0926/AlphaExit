import torch
from train_module import DQNAgent

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    for i in range(2):
        checkpoint = torch.load(f"agent{i+1}_dqn.pth", map_location=torch.device("cpu"), weights_only=True)
        model = DQNAgent(i+1)
        model.model.load_state_dict(checkpoint)
        model.model.to(device)
        torch.save(model.model.state_dict(), f"agent{i+1}_dqn.pth")