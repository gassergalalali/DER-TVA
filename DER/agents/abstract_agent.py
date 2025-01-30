"""
abstract_agent.py
"""

from abc import ABC, abstractmethod
import logging


class AbstractAgent(ABC):
    def __init__(self):
        # Set the logger
        logging.basicConfig()
        self.logger = logging.getLogger(str(self.__class__))
        self.logger.setLevel(logging.DEBUG)
        self.is_detached = False  # is detached from teh grid?

    @abstractmethod
    def setup(self):
        """Put everything that happens during setup here"""
        pass

    @abstractmethod
    def remove(self):
        """Remove self from the Simulation"""
        raise Exception(f"remove method not defined for class {self.__class__.__name__}")
