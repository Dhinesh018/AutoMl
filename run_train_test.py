from src.automl.train import train_from_config

result = train_from_config("configs/train_config.json")
print("AutoML result:", result)
