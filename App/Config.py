import json

class __AppConfig:

    def __init__(self, credentialPath="credential", configPath = "config") -> None:
        self.__config = self.readConfig(configPath)
        self.__credential = self.readCredential(credentialPath)
    
    def readCredential(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def readConfig(self, path):
        with open(path, "r") as f:
            return json.load(f)
    
    @property
    def device(self):
        return self.__config.get("device", "cpu")
    
    @property
    def serverless(self):
        return self.__config.get("serverless", 1)
    
    @property
    def hfToken(self):
        return self.__credential["huggingface"][0]
    
    @property
    def openaiToken(self):
        return self.__credential["openai"][0]

    def __str__(self) -> str:
        specified = "specified" if self.hfToken != None else "unspecified"
        return f"App Config:\ndevice: {self.device}\nserverless:{self.serverless}\ntoken:{specified}"
    

config = __AppConfig()