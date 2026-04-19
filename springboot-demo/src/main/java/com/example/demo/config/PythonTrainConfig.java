package com.example.demo.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "python.train")
public class PythonTrainConfig {

    private String pythonExecutable = "python";
    
    private String scriptPath;
    
    private String workingDirectory;
    
    private Long timeoutMinutes = 60L;
    
    private String defaultModelPath = "fno2_model.pth";
    
    private boolean redirectErrorStream = true;
}
