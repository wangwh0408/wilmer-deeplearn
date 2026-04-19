from typing import Dict, Type, Optional, Any
from train.base_trainer import BaseDeepONetTrainer, TrainingConfig, TrainingResult


class TrainerFactory:
    
    _trainers: Dict[str, Type[BaseDeepONetTrainer]] = {}
    
    @classmethod
    def register(cls, framework_name: str, trainer_class: Type[BaseDeepONetTrainer]):
        cls._trainers[framework_name.lower()] = trainer_class
    
    @classmethod
    def create(
        cls,
        framework: str,
        config: Optional[TrainingConfig] = None,
        **kwargs
    ) -> BaseDeepONetTrainer:
        framework_lower = framework.lower()
        
        if framework_lower not in cls._trainers:
            available = list(cls._trainers.keys())
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Available frameworks: {available}"
            )
        
        if config is None:
            config = TrainingConfig(**kwargs)
        elif kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = TrainingConfig.from_dict(config_dict)
        
        trainer_class = cls._trainers[framework_lower]
        return trainer_class(config)
    
    @classmethod
    def get_available_frameworks(cls) -> list:
        return list(cls._trainers.keys())
    
    @classmethod
    def is_framework_available(cls, framework: str) -> bool:
        return framework.lower() in cls._trainers


try:
    from train.pytorch_trainer import PyTorchDeepONetTrainer
    TrainerFactory.register("pytorch", PyTorchDeepONetTrainer)
    TrainerFactory.register("torch", PyTorchDeepONetTrainer)
except ImportError:
    pass

try:
    from train.paddle_trainer import PaddleDeepONetTrainer
    TrainerFactory.register("paddle", PaddleDeepONetTrainer)
    TrainerFactory.register("paddlepaddle", PaddleDeepONetTrainer)
except ImportError:
    pass


def train_deeponet(
    framework: str = "pytorch",
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> TrainingResult:
    trainer = TrainerFactory.create(framework, config, **kwargs)
    return trainer.train()


def quick_train_deeponet(
    framework: str = "pytorch",
    epochs: int = 10,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> TrainingResult:
    trainer = TrainerFactory.create(framework, config, **kwargs)
    return trainer.quick_train(epochs=epochs)
