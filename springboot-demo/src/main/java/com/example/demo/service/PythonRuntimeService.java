package com.example.demo.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
public class PythonRuntimeService {

    @Autowired
    private com.example.demo.config.PythonTrainConfig config;

    private String defaultWorkingDir;

    @PostConstruct
    public void init() {
        if (config.getWorkingDirectory() == null || config.getWorkingDirectory().isEmpty()) {
            String userDir = System.getProperty("user.dir");
            File parentDir = new File(userDir).getParentFile();
            defaultWorkingDir = parentDir != null ? parentDir.getAbsolutePath() : userDir;
            log.info("[RuntimeService] Default working directory set to: {}", defaultWorkingDir);
        } else {
            defaultWorkingDir = config.getWorkingDirectory();
        }
    }

    public com.example.demo.entity.PythonTrainResult executeWithRuntime(com.example.demo.dto.PythonTrainRequest request) {
        return executeWithRuntime(request, defaultWorkingDir);
    }

    public com.example.demo.entity.PythonTrainResult executeWithRuntime(
            com.example.demo.dto.PythonTrainRequest request, 
            String workingDir) {
        
        com.example.demo.entity.PythonTrainResult result = new com.example.demo.entity.PythonTrainResult();
        result.setStartTime(LocalDateTime.now());

        log.info("============================================================");
        log.info("[RuntimeService] Starting Python script execution with Runtime");
        log.info("============================================================");

        try {
            List<String> cmdArray = buildCommandArray(request);
            String command = String.join(" ", cmdArray);
            result.setCommand(command);

            log.info("[RuntimeService] Command: {}", command);
            log.info("[RuntimeService] Working directory: {}", workingDir);
            log.info("------------------------------------------------------------");

            File dir = new File(workingDir);
            if (!dir.exists()) {
                log.error("[RuntimeService] Working directory does not exist: {}", workingDir);
                result.setSuccess(false);
                result.setErrorMessage("Working directory does not exist: " + workingDir);
                return result;
            }

            log.info("[RuntimeService] Step 1: Calling Runtime.getRuntime().exec()");
            
            String[] cmdArrayStr = cmdArray.toArray(new String[0]);
            Process process = Runtime.getRuntime().exec(cmdArrayStr, null, dir);

            log.info("[RuntimeService] Step 2: Process started, waiting for output...");
            log.info("------------------------------------------------------------");
            log.info("[RuntimeService] === SCRIPT OUTPUT START ===");
            log.info("------------------------------------------------------------");

            StringBuilder outputBuilder = new StringBuilder();
            StringBuilder errorBuilder = new StringBuilder();
            List<String> outputLines = new ArrayList<>();
            List<String> errorLines = new ArrayList<>();

            Thread outputThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    int lineNum = 1;
                    while ((line = reader.readLine()) != null) {
                        outputLines.add(line);
                        outputBuilder.append(line).append(System.lineSeparator());
                        
                        log.info("[RuntimeService] [OUTPUT-{}] {}", 
                                String.format("%03d", lineNum), line);
                        lineNum++;
                    }
                } catch (IOException e) {
                    log.error("[RuntimeService] Error reading output stream", e);
                }
            });

            Thread errorThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
                    String line;
                    int lineNum = 1;
                    while ((line = reader.readLine()) != null) {
                        errorLines.add(line);
                        errorBuilder.append(line).append(System.lineSeparator());
                        
                        log.warn("[RuntimeService] [ERROR-{}] {}", 
                                String.format("%03d", lineNum), line);
                        lineNum++;
                    }
                } catch (IOException e) {
                    log.error("[RuntimeService] Error reading error stream", e);
                }
            });

            outputThread.start();
            if (!config.isRedirectErrorStream()) {
                errorThread.start();
            }

            log.info("[RuntimeService] Step 3: Waiting for process to complete...");
            
            boolean finished = process.waitFor(config.getTimeoutMinutes(), TimeUnit.MINUTES);

            outputThread.join();
            if (!config.isRedirectErrorStream()) {
                errorThread.join();
            }

            result.setEndTime(LocalDateTime.now());
            result.setDurationMillis(
                java.time.Duration.between(result.getStartTime(), result.getEndTime()).toMillis()
            );

            log.info("------------------------------------------------------------");
            log.info("[RuntimeService] === SCRIPT OUTPUT END ===");
            log.info("------------------------------------------------------------");

            if (!finished) {
                log.warn("[RuntimeService] Process timeout after {} minutes", config.getTimeoutMinutes());
                process.destroyForcibly();
                
                result.setSuccess(false);
                result.setExitCode(-1);
                result.setOutput(outputBuilder.toString());
                result.setErrorOutput(errorBuilder.toString());
                result.setErrorMessage("Process timeout after " + config.getTimeoutMinutes() + " minutes");
                
                log.error("[RuntimeService] Process timed out!");
            } else {
                int exitCode = process.exitValue();
                result.setExitCode(exitCode);
                result.setOutput(outputBuilder.toString());
                result.setErrorOutput(errorBuilder.toString());

                log.info("[RuntimeService] Step 4: Process completed with exit code: {}", exitCode);
                log.info("[RuntimeService] Duration: {} ms", result.getDurationMillis());

                if (exitCode == 0) {
                    result.setSuccess(true);
                    log.info("[RuntimeService] ✓ Process executed SUCCESSFULLY!");
                    
                    parseRuntimeOutput(result, outputBuilder.toString());
                    
                    log.info("------------------------------------------------------------");
                    log.info("[RuntimeService] === PARSED RESULTS ===");
                    log.info("------------------------------------------------------------");
                    log.info("[RuntimeService] Final Train Loss: {}", result.getFinalTrainLoss());
                    log.info("[RuntimeService] Final Test Loss:  {}", result.getFinalTestLoss());
                    log.info("[RuntimeService] Model Path:       {}", result.getModelPath());
                    log.info("[RuntimeService] Train Losses:     {} epochs", 
                            result.getTrainLosses() != null ? result.getTrainLosses().size() : 0);
                    
                } else {
                    result.setSuccess(false);
                    result.setErrorMessage("Process exited with code: " + exitCode);
                    log.error("[RuntimeService] ✗ Process FAILED with exit code: {}", exitCode);
                    log.error("[RuntimeService] Error output: {}", errorBuilder.toString());
                }
            }

        } catch (Exception e) {
            log.error("[RuntimeService] Exception occurred during execution", e);
            result.setSuccess(false);
            result.setErrorMessage(e.getClass().getSimpleName() + ": " + e.getMessage());
            result.setEndTime(LocalDateTime.now());
            if (result.getStartTime() != null) {
                result.setDurationMillis(
                    java.time.Duration.between(result.getStartTime(), result.getEndTime()).toMillis()
                );
            }
        }

        log.info("============================================================");
        log.info("[RuntimeService] Execution Summary:");
        log.info("  - Success: {}", result.isSuccess());
        log.info("  - Exit Code: {}", result.getExitCode());
        log.info("  - Duration: {} ms", result.getDurationMillis());
        log.info("  - Error: {}", result.getErrorMessage());
        log.info("============================================================");

        return result;
    }

    public com.example.demo.entity.PythonTrainResult quickTrainWithRuntime(int epochs) {
        log.info("[RuntimeService] Quick train requested with {} epochs", epochs);
        
        com.example.demo.dto.PythonTrainRequest request = new com.example.demo.dto.PythonTrainRequest();
        request.setQuickTrain(true);
        request.setEpochs(epochs);
        request.setNTrainSamples(Math.min(200, request.getNTrainSamples()));
        request.setNTestSamples(Math.min(100, request.getNTestSamples()));
        request.setResolution(Math.min(32, request.getResolution()));
        request.setModelSavePath(null);
        request.setLossPlotPath(null);
        request.setVerbose(true);
        
        return executeWithRuntime(request);
    }

    private List<String> buildCommandArray(com.example.demo.dto.PythonTrainRequest request) {
        List<String> cmdArray = new ArrayList<>();
        
        cmdArray.add(config.getPythonExecutable());
        cmdArray.add("-u");
        cmdArray.add("-c");
        
        StringBuilder scriptBuilder = new StringBuilder();
        
        scriptBuilder.append("import sys; sys.path.insert(0, '.'); ");
        scriptBuilder.append("from train_fno2 import train_fno2, quick_train; ");
        
        log.info("[RuntimeService] Building parameters...");
        
        if (request.getQuickTrain() != null && request.getQuickTrain()) {
            log.info("[RuntimeService] Using quick_train() function");
            scriptBuilder.append("result = quick_train(");
            appendRuntimeParam(scriptBuilder, "epochs", request.getEpochs());
            appendRuntimeParam(scriptBuilder, "n_train_samples", request.getNTrainSamples());
            appendRuntimeParam(scriptBuilder, "n_test_samples", request.getNTestSamples());
            appendRuntimeParam(scriptBuilder, "resolution", request.getResolution());
            appendRuntimeParam(scriptBuilder, "model_save_path", request.getModelSavePath());
            appendRuntimeParam(scriptBuilder, "loss_plot_path", request.getLossPlotPath());
            appendRuntimeParam(scriptBuilder, "verbose", request.getVerbose());
            scriptBuilder.append("); ");
        } else {
            log.info("[RuntimeService] Using train_fno2() function");
            scriptBuilder.append("result = train_fno2(");
            appendRuntimeParam(scriptBuilder, "modes", request.getModes());
            appendRuntimeParam(scriptBuilder, "width", request.getWidth());
            appendRuntimeParam(scriptBuilder, "epochs", request.getEpochs());
            appendRuntimeParam(scriptBuilder, "batch_size", request.getBatchSize());
            appendRuntimeParam(scriptBuilder, "learning_rate", request.getLearningRate());
            appendRuntimeParam(scriptBuilder, "weight_decay", request.getWeightDecay());
            appendRuntimeParam(scriptBuilder, "resolution", request.getResolution());
            appendRuntimeParam(scriptBuilder, "n_train_samples", request.getNTrainSamples());
            appendRuntimeParam(scriptBuilder, "n_test_samples", request.getNTestSamples());
            appendRuntimeParam(scriptBuilder, "scheduler_step_size", request.getSchedulerStepSize());
            appendRuntimeParam(scriptBuilder, "scheduler_gamma", request.getSchedulerGamma());
            appendRuntimeParam(scriptBuilder, "model_save_path", request.getModelSavePath());
            appendRuntimeParam(scriptBuilder, "loss_plot_path", request.getLossPlotPath());
            appendRuntimeParam(scriptBuilder, "verbose", request.getVerbose());
            scriptBuilder.append("); ");
        }
        
        scriptBuilder.append("print(); ");
        scriptBuilder.append("print('=' * 60); ");
        scriptBuilder.append("print('TRAIN_RESULT_START'); ");
        scriptBuilder.append("print('final_train_loss:', result.get('final_train_loss')); ");
        scriptBuilder.append("print('final_test_loss:', result.get('final_test_loss')); ");
        scriptBuilder.append("print('best_train_loss:', result.get('best_train_loss')); ");
        scriptBuilder.append("print('best_test_loss:', result.get('best_test_loss')); ");
        scriptBuilder.append("print('model_path:', result.get('model_path')); ");
        scriptBuilder.append("print('TRAIN_RESULT_END'); ");
        scriptBuilder.append("print('=' * 60); ");
        
        cmdArray.add(scriptBuilder.toString());
        
        log.info("[RuntimeService] Command built with {} parameters", 
                request.getQuickTrain() ? "quick" : "full");
        
        return cmdArray;
    }

    private void appendRuntimeParam(StringBuilder builder, String name, Object value) {
        if (value == null) {
            return;
        }
        if (builder.length() > 0 && !builder.toString().endsWith("(")) {
            builder.append(", ");
        }
        
        log.debug("[RuntimeService]   {} = {}", name, value);
        
        if (value instanceof String) {
            String strVal = (String) value;
            if (strVal.isEmpty() || "null".equalsIgnoreCase(strVal)) {
                builder.append(name).append("=None");
            } else {
                builder.append(name).append("='").append(strVal).append("'");
            }
        } else if (value instanceof Boolean) {
            builder.append(name).append("=").append((Boolean) value ? "True" : "False");
        } else if (value instanceof Double || value instanceof Float) {
            builder.append(name).append("=").append(String.format("%.10f", ((Number) value).doubleValue()));
        } else {
            builder.append(name).append("=").append(value);
        }
    }

    private void parseRuntimeOutput(com.example.demo.entity.PythonTrainResult result, String output) {
        log.info("[RuntimeService] Parsing output for results...");
        
        Pattern finalTrainLossPattern = Pattern.compile("final_train_loss:\\s*([\\d.Ee+-]+)");
        Pattern finalTestLossPattern = Pattern.compile("final_test_loss:\\s*([\\d.Ee+-]+)");
        Pattern bestTrainLossPattern = Pattern.compile("best_train_loss:\\s*([\\d.Ee+-]+)");
        Pattern bestTestLossPattern = Pattern.compile("best_test_loss:\\s*([\\d.Ee+-]+)");
        Pattern modelPathPattern = Pattern.compile("model_path:\\s*(.+)");

        Matcher matcher;

        matcher = finalTrainLossPattern.matcher(output);
        if (matcher.find()) {
            result.setFinalTrainLoss(parseDoubleSafe(matcher.group(1)));
            log.info("[RuntimeService] Parsed final_train_loss: {}", result.getFinalTrainLoss());
        } else {
            log.warn("[RuntimeService] Could not find final_train_loss in output");
        }

        matcher = finalTestLossPattern.matcher(output);
        if (matcher.find()) {
            result.setFinalTestLoss(parseDoubleSafe(matcher.group(1)));
            log.info("[RuntimeService] Parsed final_test_loss: {}", result.getFinalTestLoss());
        } else {
            log.warn("[RuntimeService] Could not find final_test_loss in output");
        }

        matcher = modelPathPattern.matcher(output);
        if (matcher.find()) {
            String path = matcher.group(1).trim();
            if (!"None".equalsIgnoreCase(path)) {
                result.setModelPath(path);
                log.info("[RuntimeService] Parsed model_path: {}", path);
            }
        }

        List<String> trainLosses = new ArrayList<>();
        List<String> testLosses = new ArrayList<>();
        
        Pattern lossPattern = Pattern.compile(
            "Epoch\\s+(\\d+)/\\d+.*?Train Loss:\\s+([\\d.Ee+-]+).*?Test Loss:\\s+([\\d.Ee+-]+)"
        );
        
        matcher = lossPattern.matcher(output);
        while (matcher.find()) {
            trainLosses.add(matcher.group(2));
            testLosses.add(matcher.group(3));
        }
        
        if (!trainLosses.isEmpty()) {
            result.setTrainLosses(trainLosses);
            log.info("[RuntimeService] Parsed {} epoch train losses", trainLosses.size());
        }
        if (!testLosses.isEmpty()) {
            result.setTestLosses(testLosses);
            log.info("[RuntimeService] Parsed {} epoch test losses", testLosses.size());
        }
    }

    private Double parseDoubleSafe(String value) {
        try {
            if (value == null || value.trim().isEmpty()) {
                return null;
            }
            String trimmed = value.trim();
            if ("None".equalsIgnoreCase(trimmed)) {
                return null;
            }
            return Double.parseDouble(trimmed);
        } catch (Exception e) {
            log.warn("[RuntimeService] Failed to parse double value: '{}'", value);
            return null;
        }
    }
}
