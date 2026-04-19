package com.example.demo.controller;

import com.example.demo.common.Result;
import com.example.demo.dto.PythonTrainRequest;
import com.example.demo.entity.PythonTrainResult;
import com.example.demo.service.PythonTrainService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/python")
public class PythonTrainController {

    @Autowired
    private PythonTrainService pythonTrainService;

    @GetMapping("/health")
    public Result<Map<String, Object>> health() {
        Map<String, Object> data = new HashMap<>();
        data.put("status", "ok");
        data.put("service", "PythonTrainService");
        data.put("timestamp", System.currentTimeMillis());
        return Result.success(data);
    }

    @PostMapping("/train")
    public Result<PythonTrainResult> train(@RequestBody PythonTrainRequest request) {
        log.info("Received training request: modes={}, epochs={}", 
                request.getModes(), request.getEpochs());
        
        PythonTrainResult result = pythonTrainService.executeTrain(request);
        
        if (result.isSuccess()) {
            return Result.success(200, "Training completed successfully", result);
        } else {
            return Result.error(500, "Training failed: " + result.getErrorMessage());
        }
    }

    @PostMapping("/quick-train/{epochs}")
    public Result<PythonTrainResult> quickTrain(@PathVariable Integer epochs) {
        log.info("Received quick training request: epochs={}", epochs);
        
        if (epochs == null || epochs <= 0) {
            return Result.error(400, "Invalid epochs value");
        }
        
        PythonTrainResult result = pythonTrainService.quickTrain(epochs);
        
        if (result.isSuccess()) {
            return Result.success(200, "Quick training completed", result);
        } else {
            return Result.error(500, "Quick training failed: " + result.getErrorMessage());
        }
    }

    @GetMapping("/status/{exitCode}")
    public Result<Map<String, String>> getStatusMessage(@PathVariable Integer exitCode) {
        Map<String, String> data = new HashMap<>();
        
        if (exitCode == 0) {
            data.put("status", "success");
            data.put("message", "Process completed successfully");
        } else if (exitCode == -1) {
            data.put("status", "timeout");
            data.put("message", "Process timeout");
        } else {
            data.put("status", "error");
            data.put("message", "Process exited with code: " + exitCode);
        }
        
        return Result.success(data);
    }

    @GetMapping("/default-config")
    public Result<PythonTrainRequest> getDefaultConfig() {
        PythonTrainRequest config = new PythonTrainRequest();
        return Result.success(config);
    }

    @PostMapping("/train/async")
    public Result<Map<String, Object>> trainAsync(@RequestBody PythonTrainRequest request) {
        log.info("Received async training request");
        
        new Thread(() -> {
            try {
                PythonTrainResult result = pythonTrainService.executeTrain(request);
                log.info("Async training completed: success={}", result.isSuccess());
            } catch (Exception e) {
                log.error("Async training error", e);
            }
        }).start();
        
        Map<String, Object> data = new HashMap<>();
        data.put("status", "started");
        data.put("message", "Training started asynchronously");
        data.put("timestamp", System.currentTimeMillis());
        
        return Result.success(202, "Training request accepted", data);
    }
}
