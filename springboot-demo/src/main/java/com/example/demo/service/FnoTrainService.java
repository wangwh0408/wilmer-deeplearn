package com.example.demo.service;

import com.example.demo.config.PythonTrainConfig;
import com.example.demo.dto.FnoTrainRequest;
import com.example.demo.entity.TrainingLogEntry;
import com.example.demo.entity.TrainingTask;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
public class FnoTrainService {

    @Autowired
    private PythonTrainConfig config;

    @Autowired
    private TrainingTaskManager taskManager;

    private String defaultWorkingDir;
    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Map<String, Process> runningProcesses = new ConcurrentHashMap<>();
    private final Map<String, Future<?>> runningFutures = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        if (config.getWorkingDirectory() == null || config.getWorkingDirectory().isEmpty()) {
            String userDir = System.getProperty("user.dir");
            File parentDir = new File(userDir).getParentFile();
            defaultWorkingDir = parentDir != null ? parentDir.getAbsolutePath() : userDir;
            log.info("[FnoTrainService] Default working directory set to: {}", defaultWorkingDir);
        } else {
            defaultWorkingDir = config.getWorkingDirectory();
        }
    }

    public String startTraining(FnoTrainRequest request) {
        return startTraining(request, defaultWorkingDir);
    }

    public String startTraining(FnoTrainRequest request, String workingDir) {
        String framework = request.getFramework();
        String taskId = taskManager.createTask(framework);
        TrainingTask task = taskManager.getTask(taskId);

        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Starting FNO training task..."));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Framework: " + framework));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Working directory: " + workingDir));

        try {
            List<String> command = buildCommand(request);
            task.setCommand(String.join(" ", command));
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Command: " + task.getCommand()));

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDir));

            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");
            
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Environment: PYTHONUNBUFFERED=1 (no buffering)"));

            task.markRunning();

            Process process = processBuilder.start();
            runningProcesses.put(taskId, process);

            Future<?> future = executorService.submit(() -> {
                try {
                    runTrainingProcess(taskId, process);
                } catch (Exception e) {
                    log.error("[FnoTrainService] Error in training process for task: {}", taskId, e);
                    TrainingTask t = taskManager.getTask(taskId);
                    if (t != null) {
                        t.addLog(TrainingLogEntry.error(t.getTaskId(), t.getNextSequence(), "Error: " + e.getMessage()));
                        t.markFailed(e.getMessage());
                    }
                } finally {
                    runningProcesses.remove(taskId);
                    runningFutures.remove(taskId);
                }
            });

            runningFutures.put(taskId, future);

            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Training started with task ID: " + taskId));
            return taskId;

        } catch (Exception e) {
            log.error("[FnoTrainService] Failed to start training task: {}", taskId, e);
            task.addLog(TrainingLogEntry.error(taskId, task.getNextSequence(), "Failed to start: " + e.getMessage()));
            task.markFailed(e.getMessage());
            return taskId;
        }
    }

    private void runTrainingProcess(String taskId, Process process) throws InterruptedException {
        TrainingTask task = taskManager.getTask(taskId);
        if (task == null) {
            process.destroyForcibly();
            return;
        }

        final TrainingTask finalTaskRef = task;
        
        Thread outputThread = new Thread(() -> {
            processStreamSynchronized(taskId, process.getInputStream(), finalTaskRef);
        });

        outputThread.setName("LogReader-" + taskId.substring(0, 8));
        outputThread.setDaemon(true);
        outputThread.start();

        try {
            boolean finished = process.waitFor(config.getTimeoutMinutes(), TimeUnit.MINUTES);
            
            outputThread.join(3000);
            
            try {
                process.getInputStream().close();
            } catch (Exception e) {
                log.warn("[FnoTrainService] Failed to close input stream: {}", e.getMessage());
            }
            try {
                process.getErrorStream().close();
            } catch (Exception e) {
                log.warn("[FnoTrainService] Failed to close error stream: {}", e.getMessage());
            }
            try {
                process.getOutputStream().close();
            } catch (Exception e) {
                log.warn("[FnoTrainService] Failed to close output stream: {}", e.getMessage());
            }

            TrainingTask finalTask = taskManager.getTask(taskId);
            if (finalTask == null) {
                return;
            }

            if (!finished) {
                process.destroyForcibly();
                process.waitFor(2, TimeUnit.SECONDS);
                
                finalTask.addLog(TrainingLogEntry.error(finalTask.getTaskId(), finalTask.getNextSequence(), 
                    "Process timeout after " + config.getTimeoutMinutes() + " minutes"));
                finalTask.setExitCode(-1);
                finalTask.markFailed("Process timeout");
                return;
            }

            int exitCode = process.exitValue();
            finalTask.setExitCode(exitCode);

            if (exitCode == 0) {
                finalTask.addLog(TrainingLogEntry.info(finalTask.getTaskId(), finalTask.getNextSequence(), 
                    "Training completed successfully!"));
                finalTask.markCompleted();
            } else {
                finalTask.addLog(TrainingLogEntry.error(finalTask.getTaskId(), finalTask.getNextSequence(), 
                    "Training failed with exit code: " + exitCode));
                finalTask.markFailed("Exit code: " + exitCode);
            }

        } catch (InterruptedException e) {
            log.warn("[FnoTrainService] Training interrupted for task: {}", taskId);
            process.destroyForcibly();
            try {
                process.waitFor(2, TimeUnit.SECONDS);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            
            TrainingTask t = taskManager.getTask(taskId);
            if (t != null) {
                t.addLog(TrainingLogEntry.warn(t.getTaskId(), t.getNextSequence(), "Training cancelled"));
                t.markCancelled();
            }
            throw e;
        }
    }

    private void processStreamSynchronized(String taskId, InputStream inputStream, TrainingTask task) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            
            String line;
            while ((line = reader.readLine()) != null) {
                TrainingTask currentTask = taskManager.getTask(taskId);
                if (currentTask == null) {
                    break;
                }

                synchronized (currentTask) {
                    int currentSeq = currentTask.getNextSequence();
                    currentTask.addLog(TrainingLogEntry.pythonOutput(currentTask.getTaskId(), currentSeq, line));
                    log.info("[FnoTrainService] [Task: {}] [Seq: {}] [Python] {}", taskId, currentSeq, line);
                    parseOutputLine(currentTask, line);
                }
            }
            
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            String remainingLine;
            try {
                while ((remainingLine = reader.readLine()) != null) {
                    TrainingTask currentTask = taskManager.getTask(taskId);
                    if (currentTask == null) {
                        break;
                    }
                    synchronized (currentTask) {
                        int currentSeq = currentTask.getNextSequence();
                        currentTask.addLog(TrainingLogEntry.pythonOutput(currentTask.getTaskId(), currentSeq, remainingLine));
                        log.info("[FnoTrainService] [Task: {}] [Seq: {}] [Python] {}", taskId, currentSeq, remainingLine);
                        parseOutputLine(currentTask, remainingLine);
                    }
                }
            } catch (Exception e) {
            }
            
        } catch (Exception e) {
            log.error("[FnoTrainService] Error reading process stream for task: {}", taskId, e);
            TrainingTask t = taskManager.getTask(taskId);
            if (t != null) {
                synchronized (t) {
                    t.addLog(TrainingLogEntry.error(t.getTaskId(), t.getNextSequence(), 
                        "Stream read error: " + e.getMessage()));
                }
            }
        }
    }

    private void parseOutputLine(TrainingTask task, String line) {
        if (line == null || line.isEmpty()) {
            return;
        }

        Pattern finalTrainLoss = Pattern.compile("Final train loss:\\s*([\\d.Ee+-]+)");
        Pattern finalTestLoss = Pattern.compile("Final test loss:\\s*([\\d.Ee+-]+)");
        Pattern bestTrainLoss = Pattern.compile("Best train loss:\\s*([\\d.Ee+-]+)");
        Pattern bestTestLoss = Pattern.compile("Best test loss:\\s*([\\d.Ee+-]+)");
        Pattern modelPath = Pattern.compile("Model saved to:\\s*(.+)");

        Matcher matcher;

        matcher = finalTrainLoss.matcher(line);
        if (matcher.find()) {
            task.setFinalTrainLoss(parseDouble(matcher.group(1)));
        }

        matcher = finalTestLoss.matcher(line);
        if (matcher.find()) {
            task.setFinalTestLoss(parseDouble(matcher.group(1)));
        }

        matcher = bestTrainLoss.matcher(line);
        if (matcher.find()) {
            task.setBestTrainLoss(parseDouble(matcher.group(1)));
        }

        matcher = bestTestLoss.matcher(line);
        if (matcher.find()) {
            task.setBestTestLoss(parseDouble(matcher.group(1)));
        }

        matcher = modelPath.matcher(line);
        if (matcher.find()) {
            task.setModelPath(matcher.group(1).trim());
        }
    }

    private Double parseDouble(String value) {
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
            return null;
        }
    }

    public boolean cancelTraining(String taskId) {
        Process process = runningProcesses.get(taskId);
        Future<?> future = runningFutures.get(taskId);

        if (process != null && process.isAlive()) {
            try {
                process.destroy();
                if (!process.waitFor(3, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                }
            } catch (InterruptedException e) {
                process.destroyForcibly();
                Thread.currentThread().interrupt();
            }
            
            runningProcesses.remove(taskId);

            if (future != null) {
                future.cancel(true);
                runningFutures.remove(taskId);
            }

            TrainingTask task = taskManager.getTask(taskId);
            if (task != null) {
                synchronized (task) {
                    task.addLog(TrainingLogEntry.warn(task.getTaskId(), task.getNextSequence(), 
                        "Training cancelled by user"));
                }
                task.markCancelled();
            }

            log.info("[FnoTrainService] Training cancelled for task: {}", taskId);
            return true;
        }

        return false;
    }

    public TrainingTask getTaskStatus(String taskId) {
        return taskManager.getTask(taskId);
    }

    public List<TrainingLogEntry> getLogs(String taskId, Integer since) {
        TrainingTask task = taskManager.getTask(taskId);
        if (task == null) {
            return new ArrayList<>();
        }

        if (since == null || since < 0) {
            return new ArrayList<>(task.getLogs());
        }

        return task.getLogsSince(since);
    }

    public List<TrainingLogEntry> getLogsBySequence(String taskId, Integer sinceSequence) {
        TrainingTask task = taskManager.getTask(taskId);
        if (task == null) {
            return new ArrayList<>();
        }

        if (sinceSequence == null || sinceSequence < 0) {
            return new ArrayList<>(task.getLogs());
        }

        return task.getLogsSinceSequence(sinceSequence);
    }

    public List<TrainingLogEntry> getNewLogs(String taskId) {
        TrainingTask task = taskManager.getTask(taskId);
        if (task == null) {
            return new ArrayList<>();
        }

        return task.getNewLogsBySequence();
    }

    public Integer getCurrentMaxSequence(String taskId) {
        TrainingTask task = taskManager.getTask(taskId);
        if (task == null) {
            return -1;
        }
        return task.getMaxSequence();
    }

    private List<String> buildCommand(FnoTrainRequest request) {
        List<String> command = new ArrayList<>();

        command.add(config.getPythonExecutable());
        command.add("-u");

        String scriptName;
        if ("paddle".equalsIgnoreCase(request.getFramework()) || 
            "paddlepaddle".equalsIgnoreCase(request.getFramework())) {
            scriptName = "train_fno_cfdbench_paddle.py";
        } else {
            scriptName = "train_fno_cfdbench.py";
        }

        command.add(scriptName);

        if (request.getDataRoot() != null && !request.getDataRoot().isEmpty()) {
            command.add("--data_root");
            command.add(request.getDataRoot());
        }

        if (request.getProblems() != null && !request.getProblems().isEmpty()) {
            command.add("--problems");
            command.addAll(request.getProblems());
        }

        if (request.getCategories() != null && !request.getCategories().isEmpty()) {
            command.add("--categories");
            command.addAll(request.getCategories());
        }

        if (request.getMaxCases() != null) {
            command.add("--max_cases");
            command.add(request.getMaxCases().toString());
        }

        if (request.getModes1() != null) {
            command.add("--modes1");
            command.add(request.getModes1().toString());
        }

        if (request.getModes2() != null) {
            command.add("--modes2");
            command.add(request.getModes2().toString());
        }

        if (request.getWidth() != null) {
            command.add("--width");
            command.add(request.getWidth().toString());
        }

        if (request.getNLayers() != null) {
            command.add("--n_layers");
            command.add(request.getNLayers().toString());
        }

        if (request.getHiddenDim() != null) {
            command.add("--hidden_dim");
            command.add(request.getHiddenDim().toString());
        }

        if (request.getLearningRate() != null) {
            command.add("--learning_rate");
            command.add(request.getLearningRate().toString());
        }

        if (request.getWeightDecay() != null) {
            command.add("--weight_decay");
            command.add(request.getWeightDecay().toString());
        }

        if (request.getBatchSize() != null) {
            command.add("--batch_size");
            command.add(request.getBatchSize().toString());
        }

        if (request.getEpochs() != null) {
            command.add("--epochs");
            command.add(request.getEpochs().toString());
        }

        if (request.getSchedulerStepSize() != null) {
            command.add("--scheduler_step_size");
            command.add(request.getSchedulerStepSize().toString());
        }

        if (request.getSchedulerGamma() != null) {
            command.add("--scheduler_gamma");
            command.add(request.getSchedulerGamma().toString());
        }

        if (request.getInputSteps() != null) {
            command.add("--input_steps");
            command.add(request.getInputSteps().toString());
        }

        if (request.getOutputSteps() != null) {
            command.add("--output_steps");
            command.add(request.getOutputSteps().toString());
        }

        if (request.getTrainRatio() != null) {
            command.add("--train_ratio");
            command.add(request.getTrainRatio().toString());
        }

        if (request.getSeed() != null) {
            command.add("--seed");
            command.add(request.getSeed().toString());
        }

        if (request.getModelSavePath() != null && !request.getModelSavePath().isEmpty()) {
            command.add("--model_save_path");
            command.add(request.getModelSavePath());
        }

        if (request.getLogFile() != null && !request.getLogFile().isEmpty()) {
            command.add("--log_file");
            command.add(request.getLogFile());
        }

        if (request.getVerbose() != null && request.getVerbose()) {
            command.add("--verbose");
        }

        return command;
    }

    public List<TrainingTask> getAllTasks() {
        return new ArrayList<>(taskManager.getAllTasks().values());
    }

    public String quickTestTraining() {
        FnoTrainRequest request = new FnoTrainRequest();
        request.setFramework("pytorch");
        request.setEpochs(3);
        request.setMaxCases(5);
        request.setWidth(16);
        request.setNLayers(2);
        request.setVerbose(true);
        return startTraining(request);
    }

    public String quickTestTrainingPaddle() {
        FnoTrainRequest request = new FnoTrainRequest();
        request.setFramework("paddle");
        request.setEpochs(3);
        request.setMaxCases(5);
        request.setWidth(16);
        request.setNLayers(2);
        request.setVerbose(true);
        return startTraining(request);
    }
}
