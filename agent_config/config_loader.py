"""
Configuration Loader
Loads LLM and system configuration from config.yaml
"""
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration manager for SPC system"""
    
    def __init__(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Default to config.yaml in the same directory as this file
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(config_path)
        
        # Load environment variables from project root
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        
        # Load YAML configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration if file doesn't exist
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration if config.yaml doesn't exist"""
        return {
            'llm': {
                'provider': 'groq',
                'model': 'llama-3.1-8b-instant'
            },
            'agents': {
                'temperature': 0.0,
                'max_tokens': None,
                'timeout': 60
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'cors_origins': ['*']
            },
            'uploads': {
                'temp_directory': 'temp_uploads',
                'max_file_size_mb': 10,
                'allowed_extensions': ['.csv', '.xlsx', '.txt']
            },
            'reports': {
                'auto_generate': True,
                'save_with_data': True,
                'include_plots': True
            }
        }
    
    @property
    def llm_provider(self):
        """Get LLM provider (e.g., 'groq', 'openai')"""
        return self.config.get('llm', {}).get('provider', 'groq')
    
    @property
    def llm_model(self):
        """Get LLM model name (e.g., 'llama-3.1-8b-instant')"""
        return self.config.get('llm', {}).get('model', 'llama-3.1-8b-instant')
    
    @property
    def llm_model_string(self):
        """Get full LLM model string (e.g., 'groq:llama-3.1-8b-instant')"""
        return f"{self.llm_provider}:{self.llm_model}"
    
    @property
    def llm_api_key(self):
        """Get API key for the configured LLM provider"""
        provider = self.llm_provider.upper()
        key_name = f"{provider}_API_KEY"
        api_key = os.getenv(key_name)
        
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{self.llm_provider}'. "
                f"Please set {key_name} in your .env file. "
                f"See env.example for instructions."
            )
        
        return api_key
    
    @property
    def agent_temperature(self):
        """Get agent temperature setting"""
        return self.config.get('agents', {}).get('temperature', 0.0)
    
    @property
    def agent_max_tokens(self):
        """Get max tokens setting"""
        return self.config.get('agents', {}).get('max_tokens', None)
    
    @property
    def agent_timeout(self):
        """Get request timeout"""
        return self.config.get('agents', {}).get('timeout', 60)
    
    @property
    def api_host(self):
        """Get API host"""
        return self.config.get('api', {}).get('host', '0.0.0.0')
    
    @property
    def api_port(self):
        """Get API port"""
        return self.config.get('api', {}).get('port', 8000)
    
    @property
    def cors_origins(self):
        """Get CORS origins"""
        return self.config.get('api', {}).get('cors_origins', ['*'])
    
    @property
    def temp_upload_dir(self):
        """Get temporary upload directory"""
        return self.config.get('uploads', {}).get('temp_directory', 'temp_uploads')
    
    def __repr__(self):
        """String representation of config"""
        return f"Config(provider={self.llm_provider}, model={self.llm_model})"


# Global config instance
config = Config()


# Convenience functions
def get_llm_model_string():
    """Get the full LLM model string (provider:model)"""
    return config.llm_model_string


def get_api_key():
    """Get API key for configured provider"""
    return config.llm_api_key


def get_config():
    """Get the global config instance"""
    return config

