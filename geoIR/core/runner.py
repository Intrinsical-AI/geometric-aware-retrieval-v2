#!/usr/bin/env python3
"""
Motor de Experimentos Genérico
==============================

Este módulo proporciona una clase base `ExperimentRunner` para estandarizar
la ejecución de experimentos, el logging y la gestión de resultados.
"""

import abc
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ExperimentRunner(abc.ABC):
    """Clase base abstracta para ejecutar un experimento de forma estandarizada."""

    def __init__(self, experiment_name: str, base_output_dir: str = "experiments/results"):
        """
        Args:
            experiment_name: Nombre descriptivo del experimento (p.ej., 'fiqa_experiment').
            base_output_dir: Directorio raíz donde se guardarán todos los resultados.
        """
        self.experiment_name = experiment_name
        self.base_output_dir = Path(base_output_dir)
        self.run_dir: Path = None
        self.logger: logging.Logger = None
        self.config: Dict[str, Any] = {}

    def _setup_run_environment(self):
        """Crea un directorio único para la ejecución y configura el logging."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.base_output_dir / self.experiment_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Configurar logging para guardar en un archivo y mostrar en consola
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Evitar duplicar handlers si se llama varias veces
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Handler para el archivo
        file_handler = logging.FileHandler(self.run_dir / "run.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Handler para la consola
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.logger.addHandler(stream_handler)

        self.logger.info(f"Resultados de la ejecución en: {self.run_dir}")

    def _save_config(self, config: Any):
        """Guarda la configuración del experimento en un archivo JSON."""
        if not self.run_dir:
            raise RuntimeError("El entorno de ejecución no ha sido inicializado. Llama a _setup_run_environment() primero.")

        config_path = self.run_dir / "config.json"
        
        config_dict = {}
        if is_dataclass(config):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            self.logger.warning("La configuración no es un dataclass o un diccionario, se guardará como string.")
            config_dict = {'config': str(config)}

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info(f"Configuración guardada en {config_path}")

    @abc.abstractmethod
    def run(self):
        """Método abstracto que debe ser implementado por cada experimento.
        
        Aquí es donde se define la lógica principal del experimento.
        """
        pass

    def start(self, config: Any):
        """Punto de entrada para iniciar el experimento."""
        # Guardar la configuración internamente
        if is_dataclass(config):
            self.config = asdict(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = {'config': str(config)}

        # Configurar el entorno de ejecución y guardar la configuración en un archivo
        self._setup_run_environment()
        self._save_config(self.config)
        
        try:
            # Ejecutar la lógica principal del experimento
            self.run()
        except Exception as e:
            self.logger.error("El experimento ha fallado con una excepción.", exc_info=True)
            raise e
        
        self.logger.info("El experimento ha finalizado con éxito.")

    def save_results(self, results: list[dict], filename: str) -> Path:
        """Guarda los resultados en formato JSON.
        
        Args:
            results: Lista de diccionarios con los resultados.
            filename: Nombre del archivo de salida.
            
        Returns:
            Path al archivo guardado.
        """
        if not self.run_dir:
            raise RuntimeError("El entorno de ejecución no ha sido inicializado.")
            
        path = self.run_dir / filename
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Resultados guardados en {path}")
        return path

    def save_dataframe(self, df, filename: str) -> Path:
        """Guarda un DataFrame en formato CSV.
        
        Args:
            df: DataFrame de pandas a guardar.
            filename: Nombre del archivo de salida.
            
        Returns:
            Path al archivo guardado.
        """
        if not self.run_dir:
            raise RuntimeError("El entorno de ejecución no ha sido inicializado.")
            
        path = self.run_dir / filename
        df.to_csv(path, index=False)
        self.logger.info(f"DataFrame guardado en {path}")
        return path
