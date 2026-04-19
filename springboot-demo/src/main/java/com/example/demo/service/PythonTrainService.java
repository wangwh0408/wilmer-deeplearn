package com.example.demo.service;

import com.example.demo.config.PythonTrainConfig;
import com.example.demo.dto.PythonTrainRequest;
import com.example.demo.entity.PythonTrainResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
public class PythonTrainService {

    @Autowired
    private PythonTrainConfig config;

    private String defaultWorkingDir;

    @PostConstruct
    public void init() {
        if (config.getWorkingDirectory() == null || config.getWorkingDirectory().isEmpty()) {
            String userDir = System.getProperty("user.dir");
            File parentDir = new File(userDir).getParentFile();
            defaultWorkingDir = parentDir != null ? parentDir.getAbsolutePath() : userDir;
            log.info("Default working directory set to: {}", defaultWorkingDir);
        } else {
            defaultWorkingDir = config.getWorkingDirectory();
        }
    }

    public PythonTrainResult executeTrain(PythonTrainRequest request) {
        return executeTrain(request, defaultWorkingDir);
    }

    public PythonTrainResult executeTrain(PythonTrainRequest request, String workingDir) {
        PythonTrainResult result = new PythonTrainResult();
        result.setStartTime(LocalDateTime.now());

        List<String> command = buildCommand(request);
        result.setCommand(String.join(" ", command));

        log.info("Starting Python training with command: {}", result.getCommand());
        log.info("Working directory: {}", workingDir);

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(new File(workingDir));
        
        if (config.isRedirectErrorStream()) {
            processBuilder.redirectErrorStream(true);
        }

        Process process = null;
        StringBuilder outputBuilder = new StringBuilder();
        StringBuilder errorBuilder = new StringBuilder();

        try {
            process = processBuilder.start();
            
            final Process finalProcess = process;
            
            Thread outputThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(finalProcess.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        outputBuilder.append(line).append(System.lineSeparator());
                        log.info("[Python Output] {}", line);
                    }
                } catch (Exception e) {
                    log.error("Error reading process output", e);
                }
            });

            Thread errorThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(finalProcess.getErrorStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        errorBuilder.append(line).append(System.lineSeparator());
                        log.error("[Python Error] {}", line);
                    }
                } catch (Exception e) {
                    log.error("Error reading process error stream", e);
                }
            });

            outputThread.start();
            if (!config.isRedirectErrorStream()) {
                errorThread.start();
            }

            boolean finished = process.waitFor(config.getTimeoutMinutes(), TimeUnit.MINUTES);
            
            outputThread.join();
            if (!config.isRedirectErrorStream()) {
                errorThread.join();
            }

            result.setEndTime(LocalDateTime.now());
            result.setDurationMillis(
                java.time.Duration.between(result.getStartTime(), result.getEndTime()).toMillis()
            );

            if (!finished) {
                log.warn("Process timeout after {} minutes", config.getTimeoutMinutes());
                process.destroyForcibly();
                result.setSuccess(false);
                result.setExitCode(-1);
                result.setErrorMessage("Process timeout after " + config.getTimeoutMinutes() + " minutes");
            } else {
                int exitCode = process.exitValue();
                result.setExitCode(exitCode);
                result.setSuccess(exitCode == 0);
                result.setOutput(outputBuilder.toString());
                result.setErrorOutput(errorBuilder.toString());

                if (exitCode == 0) {
                    parseOutput(result, outputBuilder.toString());
                    log.info("Python training completed successfully in {} ms", result.getDurationMillis());
                } else {
                    result.setErrorMessage("Process exited with code: " + exitCode);
                    log.error("Python training failed with exit code: {}", exitCode);
                }
            }

        } catch (Exception e) {
            log.error("Error executing Python training", e);
            result.setSuccess(false);
            result.setErrorMessage(e.getMessage());
            result.setEndTime(LocalDateTime.now());
            if (result.getStartTime() != null) {
                result.setDurationMillis(
                    java.time.Duration.between(result.getStartTime(), result.getEndTime()).toMillis()
                );
            }
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
            }
        }

        return result;
    }

    public PythonTrainResult quickTrain(int epochs) {
        PythonTrainRequest request = new PythonTrainRequest();
        request.setQuickTrain(true);
        request.setEpochs(epochs);
        request.setNTrainSamples(Math.min(200, request.getNTrainSamples()));
        request.setNTestSamples(Math.min(100, request.getNTestSamples()));
        request.setResolution(Math.min(32, request.getResolution()));
        request.setModelSavePath(null);
        request.setLossPlotPath(null);
        return executeTrain(request);
    }

    private List<String> buildCommand(PythonTrainRequest request) {
        List<String> command = new ArrayList<>();
        
        command.add(config.getPythonExecutable());
        
        command.add("-c");
        
        StringBuilder scriptBuilder = new StringBuilder();
        scriptBuilder.append("import sys; sys.path.insert(0, '.'); ");
        scriptBuilder.append("from train_fno2 import train_fno2, quick_train; ");
        
        if (request.getQuickTrain() != null && request.getQuickTrain()) {
            scriptBuilder.append("result = quick_train(");
            appendParam(scriptBuilder, "epochs", request.getEpochs());
            appendParam(scriptBuilder, "n_train_samples", request.getNTrainSamples());
            appendParam(scriptBuilder, "n_test_samples", request.getNTestSamples());
            appendParam(scriptBuilder, "resolution", request.getResolution());
            appendParam(scriptBuilder, "model_save_path", request.getModelSavePath());
            appendParam(scriptBuilder, "loss_plot_path", request.getLossPlotPath());
            appendParam(scriptBuilder, "verbose", request.getVerbose());
            scriptBuilder.append("); ");
        } else {
            scriptBuilder.append("result = train_fno2(");
            appendParam(scriptBuilder, "modes", request.getModes());
            appendParam(scriptBuilder, "width", request.getWidth());
            appendParam(scriptBuilder, "epochs", request.getEpochs());
            appendParam(scriptBuilder, "batch_size", request.getBatchSize());
            appendParam(scriptBuilder, "learning_rate", request.getLearningRate());
            appendParam(scriptBuilder, "weight_decay", request.getWeightDecay());
            appendParam(scriptBuilder, "resolution", request.getResolution());
            appendParam(scriptBuilder, "n_train_samples", request.getNTrainSamples());
            appendParam(scriptBuilder, "n_test_samples", request.getNTestSamples());
            appendParam(scriptBuilder, "scheduler_step_size", request.getSchedulerStepSize());
            appendParam(scriptBuilder, "scheduler_gamma", request.getSchedulerGamma());
            appendParam(scriptBuilder, "model_save_path", request.getModelSavePath());
            appendParam(scriptBuilder, "loss_plot_path", request.getLossPlotPath());
            appendParam(scriptBuilder, "verbose", request.getVerbose());
            scriptBuilder.append("); ");
        }
        
        scriptBuilder.append("print('TRAIN_RESULT_START'); ");
        scriptBuilder.append("print('final_train_loss:', result.get('final_train_loss')); ");
        scriptBuilder.append("print('final_test_loss:', result.get('final_test_loss')); ");
        scriptBuilder.append("print('best_train_loss:', result.get('best_train_loss')); ");
        scriptBuilder.append("print('best_test_loss:', result.get('best_test_loss')); ");
        scriptBuilder.append("print('model_path:', result.get('model_path')); ");
        scriptBuilder.append("print('TRAIN_RESULT_END')");
        
        command.add(scriptBuilder.toString());
        
        return command;
    }

    private void appendParam(StringBuilder builder, String name, Object value) {
        if (value == null) {
            return;
        }
        if (builder.length() > 0 && !builder.toString().endsWith("(")) {
            builder.append(", ");
        }
        if (value instanceof String) {
            if (((String) value).isEmpty() || "null".equalsIgnoreCase((String) value)) {
                builder.append(name).append("=None");
            } else {
                builder.append(name).append("='").append(value).append("'");
            }
        } else if (value instanceof Boolean) {
            builder.append(name).append("=").append((Boolean) value ? "True" : "False");
        } else {
            builder.append(name).append("=").append(value);
        }
    }

    private void parseOutput(PythonTrainResult result, String output) {
        Pattern finalTrainLossPattern = Pattern.compile("final_train_loss:\\s*([\\d.Ee+-]+)");
        Pattern finalTestLossPattern = Pattern.compile("final_test_loss:\\s*([\\d.Ee+-]+)");
        Pattern bestTrainLossPattern = Pattern.compile("best_train_loss:\\s*([\\d.Ee+-]+)");
        Pattern bestTestLossPattern = Pattern.compile("best_test_loss:\\s*([\\d.Ee+-]+)");
        Pattern modelPathPattern = Pattern.compile("model_path:\\s*(.+)");

        Matcher matcher;

        matcher = finalTrainLossPattern.matcher(output);
        if (matcher.find()) {
            result.setFinalTrainLoss(parseDouble(matcher.group(1)));
        }

        matcher = finalTestLossPattern.matcher(output);
        if (matcher.find()) {
            result.setFinalTestLoss(parseDouble(matcher.group(1)));
        }

        matcher = modelPathPattern.matcher(output);
        if (matcher.find()) {
            String path = matcher.group(1).trim();
            if (!"None".equalsIgnoreCase(path)) {
                result.setModelPath(path);
            }
        }

        List<String> trainLosses = new ArrayList<>();
        List<String> testLosses = new ArrayList<>();
        
        Pattern lossPattern = Pattern.compile(
            "Epoch\\s+(\\d+)/\\d+\\s+\\|\\s+" +
            "Train Loss:\\s+([\\d.Ee+-]+)\\s+\\|\\s+" +
            "Test Loss:\\s+([\\d.Ee+-]+)"
        );
        
        matcher = lossPattern.matcher(output);
        while (matcher.find()) {
            trainLosses.add(matcher.group(2));
            testLosses.add(matcher.group(3));
        }
        
        if (!trainLosses.isEmpty()) {
            result.setTrainLosses(trainLosses);
        }
        if (!testLosses.isEmpty()) {
            result.setTestLosses(testLosses);
        }
    }

    private Double parseDouble(String value) {
        try {
            if ("None".equalsIgnoreCase(value.trim())) {
                return null;
            }
            return Double.parseDouble(value.trim());
        } catch (Exception e) {
            log.warn("Failed to parse double value: {}", value);
            return null;
        }
    }
}
