from vmoe import app
from vmoe.train import trainer

if __name__ == '__main__':
  app.run(trainer.train_and_evaluate)