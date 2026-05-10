package com.example.demo.service;

import com.example.demo.config.PythonTrainConfig;
import com.example.demo.dto.Fno2EnhancedTrainRequest;
import com.example.demo.dto.Fno2EvaluateRequest;
import com.example.demo.dto.Fno2PaddleTrainRequest;
import com.example.demo.dto.FnoTrainRequest;
import com.example.demo.entity.TrainingLogEntry;
import com.example.demo.entity.TrainingLogEntryEntity;
import com.example.demo.entity.TrainingTask;
import com.example.demo.entity.TrainingTaskEntity;
import com.example.demo.repository.TrainingLogEntryRepository;
import com.example.demo.repository.TrainingTaskRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.transaction.Transactional;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
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

    @Autowired
    private TrainingTaskRepository trainingTaskRepository;

    @Autowired
    private TrainingLogEntryRepository trainingLogEntryRepository;

    private String defaultWorkingDir;
    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Map<String, Process> runningProcesses = new ConcurrentHashMap<>();
    private final Map<String, Future<?>> runningFutures = new ConcurrentHashMap<>();

    private final Map<String, LinkedBlockingQueue<TrainingLogEntryEntity>> logQueues = new ConcurrentHashMap<>();
    private final ScheduledExecutorService logFlushScheduler = Executors.newSingleThreadScheduledExecutor();
    private static final int LOG_BATCH_SIZE = 50;
    private static final int LOG_FLUSH_INTERVAL_MS = 1000;

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
        
        logFlushScheduler.scheduleAtFixedRate(this::flushAllLogQueues, 
            LOG_FLUSH_INTERVAL_MS, LOG_FLUSH_INTERVAL_MS, TimeUnit.MILLISECONDS);
        log.info("[FnoTrainService] Log flush scheduler started with interval: {}ms", LOG_FLUSH_INTERVAL_MS);
    }

    @Transactional
    public TrainingTaskEntity createTaskEntity(String taskId, String command, 
            TrainingTaskEntity.TaskType taskType, String framework) {
        TrainingTaskEntity entity = new TrainingTaskEntity();
        entity.setTaskId(taskId);
        entity.setStatus(TrainingTaskEntity.Status.PENDING);
        entity.setTaskType(taskType);
        entity.setFramework(framework);
        entity.setCommand(command);
        entity.setLogCount(0);
        return trainingTaskRepository.save(entity);
    }

    @Transactional
    public void updateTaskStatus(String taskId, TrainingTaskEntity.Status status) {
        Optional<TrainingTaskEntity> optional = trainingTaskRepository.findByTaskId(taskId);
        if (optional.isPresent()) {
            TrainingTaskEntity entity = optional.get();
            if (status == TrainingTaskEntity.Status.RUNNING) {
                entity.markRunning();
            } else if (status == TrainingTaskEntity.Status.COMPLETED) {
                entity.markCompleted();
            } else if (status == TrainingTaskEntity.Status.FAILED) {
                entity.markFailed(null);
            } else if (status == TrainingTaskEntity.Status.CANCELLED) {
                entity.markCancelled();
            }
            trainingTaskRepository.save(entity);
        }
    }

    @Transactional
    public void updateTaskEntity(String taskId, Integer exitCode, String errorMessage,
            Double finalTrainLoss, Double finalTestLoss, Double bestTestLoss, String modelPath) {
        Optional<TrainingTaskEntity> optional = trainingTaskRepository.findByTaskId(taskId);
        if (optional.isPresent()) {
            TrainingTaskEntity entity = optional.get();
            if (exitCode != null) {
                entity.setExitCode(exitCode);
            }
            if (errorMessage != null) {
                entity.setErrorMessage(errorMessage);
            }
            if (finalTrainLoss != null) {
                entity.setFinalTrainLoss(finalTrainLoss);
            }
            if (finalTestLoss != null) {
                entity.setFinalTestLoss(finalTestLoss);
            }
            if (bestTestLoss != null) {
                entity.setBestTestLoss(bestTestLoss);
            }
            if (modelPath != null) {
                entity.setModelPath(modelPath);
            }
            trainingTaskRepository.save(entity);
        }
    }

    public void addLogToQueue(String taskId, TrainingLogEntryEntity logEntry) {
        LinkedBlockingQueue<TrainingLogEntryEntity> queue = logQueues.computeIfAbsent(
            taskId, k -> new LinkedBlockingQueue<>()
        );
        queue.offer(logEntry);
        
        if (queue.size() >= LOG_BATCH_SIZE) {
            flushLogQueue(taskId);
        }
    }

    @Transactional
    public void flushLogQueue(String taskId) {
        LinkedBlockingQueue<TrainingLogEntryEntity> queue = logQueues.get(taskId);
        if (queue == null || queue.isEmpty()) {
            return;
        }
        
        List<TrainingLogEntryEntity> entries = new ArrayList<>();
        queue.drainTo(entries);
        
        if (!entries.isEmpty()) {
            trainingLogEntryRepository.saveAll(entries);
            
            Optional<TrainingTaskEntity> optional = trainingTaskRepository.findByTaskId(taskId);
            if (optional.isPresent()) {
                TrainingTaskEntity entity = optional.get();
                int currentCount = entity.getLogCount() != null ? entity.getLogCount() : 0;
                entity.setLogCount(currentCount + entries.size());
                trainingTaskRepository.save(entity);
            }
            
            log.debug("[FnoTrainService] Flushed {} log entries for task: {}", entries.size(), taskId);
        }
    }

    public void flushAllLogQueues() {
        for (String taskId : logQueues.keySet()) {
            flushLogQueue(taskId);
        }
    }

    @Transactional
    public void saveLogImmediate(String taskId, TrainingLogEntryEntity logEntry) {
        trainingLogEntryRepository.save(logEntry);
        
        Optional<TrainingTaskEntity> optional = trainingTaskRepository.findByTaskId(taskId);
        if (optional.isPresent()) {
            TrainingTaskEntity entity = optional.get();
            entity.incrementLogCount();
            trainingTaskRepository.save(entity);
        }
    }

    public List<TrainingLogEntryEntity> getLogsFromDatabase(String taskId) {
        return trainingLogEntryRepository.findByTaskIdOrderBySequenceAsc(taskId);
    }

    public List<TrainingLogEntryEntity> getNewLogsFromDatabase(String taskId, Integer lastSequence) {
        if (lastSequence == null) {
            return trainingLogEntryRepository.findByTaskIdOrderBySequenceAsc(taskId);
        }
        return trainingLogEntryRepository.findByTaskIdAndSequenceGreaterThanOrderBySequenceAsc(taskId, lastSequence);
    }

    public Optional<TrainingTaskEntity> getTaskEntity(String taskId) {
        return trainingTaskRepository.findByTaskId(taskId);
    }

    public List<TrainingTaskEntity> getAllTaskEntities() {
        return trainingTaskRepository.findAllByOrderByCreateTimeDesc();
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
            String commandStr = String.join(" ", command);
            task.setCommand(commandStr);
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Command: " + commandStr));
            
            createTaskEntity(taskId, commandStr, 
                TrainingTaskEntity.TaskType.TRAINING, framework);

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDir));

            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");
            
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Environment: PYTHONUNBUFFERED=1 (no buffering)"));

            task.markRunning();
            updateTaskStatus(taskId, TrainingTaskEntity.Status.RUNNING);

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
                    flushLogQueue(taskId);
                }
            });

            runningFutures.put(taskId, future);

            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Training started with task ID: " + taskId));
            return taskId;

        } catch (Exception e) {
            log.error("[FnoTrainService] Failed to start training task: {}", taskId, e);
            task.addLog(TrainingLogEntry.error(taskId, task.getNextSequence(), "Failed to start: " + e.getMessage()));
            task.markFailed(e.getMessage());
            updateTaskStatus(taskId, TrainingTaskEntity.Status.FAILED);
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
                flushLogQueue(taskId);
                return;
            }

            if (!finished) {
                process.destroyForcibly();
                process.waitFor(2, TimeUnit.SECONDS);
                
                int currentSeq = finalTask.getNextSequence();
                TrainingLogEntry logEntry = TrainingLogEntry.error(
                    finalTask.getTaskId(), currentSeq, 
                    "Process timeout after " + config.getTimeoutMinutes() + " minutes");
                finalTask.addLog(logEntry);
                finalTask.setExitCode(-1);
                finalTask.markFailed("Process timeout");
                
                TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                    finalTask.getTaskId(), logEntry);
                addLogToQueue(taskId, logEntity);
                flushLogQueue(taskId);
                
                updateTaskStatus(taskId, TrainingTaskEntity.Status.FAILED);
                updateTaskEntity(taskId, -1, "Process timeout", 
                    finalTask.getFinalTrainLoss(), finalTask.getFinalTestLoss(), 
                    finalTask.getBestTestLoss(), finalTask.getModelPath());
                return;
            }

            int exitCode = process.exitValue();
            finalTask.setExitCode(exitCode);

            if (exitCode == 0) {
                int currentSeq = finalTask.getNextSequence();
                TrainingLogEntry logEntry = TrainingLogEntry.info(
                    finalTask.getTaskId(), currentSeq, "Training completed successfully!");
                finalTask.addLog(logEntry);
                finalTask.markCompleted();
                
                TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                    finalTask.getTaskId(), logEntry);
                addLogToQueue(taskId, logEntity);
                flushLogQueue(taskId);
                
                updateTaskStatus(taskId, TrainingTaskEntity.Status.COMPLETED);
                updateTaskEntity(taskId, exitCode, null, 
                    finalTask.getFinalTrainLoss(), finalTask.getFinalTestLoss(), 
                    finalTask.getBestTestLoss(), finalTask.getModelPath());
            } else {
                int currentSeq = finalTask.getNextSequence();
                TrainingLogEntry logEntry = TrainingLogEntry.error(
                    finalTask.getTaskId(), currentSeq, 
                    "Training failed with exit code: " + exitCode);
                finalTask.addLog(logEntry);
                finalTask.markFailed("Exit code: " + exitCode);
                
                TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                    finalTask.getTaskId(), logEntry);
                addLogToQueue(taskId, logEntity);
                flushLogQueue(taskId);
                
                updateTaskStatus(taskId, TrainingTaskEntity.Status.FAILED);
                updateTaskEntity(taskId, exitCode, "Exit code: " + exitCode, 
                    finalTask.getFinalTrainLoss(), finalTask.getFinalTestLoss(), 
                    finalTask.getBestTestLoss(), finalTask.getModelPath());
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
                int currentSeq = t.getNextSequence();
                TrainingLogEntry logEntry = TrainingLogEntry.warn(
                    t.getTaskId(), currentSeq, "Training cancelled");
                t.addLog(logEntry);
                t.markCancelled();
                
                TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                    t.getTaskId(), logEntry);
                addLogToQueue(taskId, logEntity);
                flushLogQueue(taskId);
                
                updateTaskStatus(taskId, TrainingTaskEntity.Status.CANCELLED);
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
                    TrainingLogEntry logEntry = TrainingLogEntry.pythonOutput(
                        currentTask.getTaskId(), currentSeq, line);
                    currentTask.addLog(logEntry);
                    
                    TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                        currentTask.getTaskId(), logEntry);
                    addLogToQueue(taskId, logEntity);
                    
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
                        TrainingLogEntry logEntry = TrainingLogEntry.pythonOutput(
                            currentTask.getTaskId(), currentSeq, remainingLine);
                        currentTask.addLog(logEntry);
                        
                        TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                            currentTask.getTaskId(), logEntry);
                        addLogToQueue(taskId, logEntity);
                        
                        log.info("[FnoTrainService] [Task: {}] [Seq: {}] [Python] {}", taskId, currentSeq, remainingLine);
                        parseOutputLine(currentTask, remainingLine);
                    }
                }
            } catch (Exception e) {
            }
            
            flushLogQueue(taskId);
            
        } catch (Exception e) {
            log.error("[FnoTrainService] Error reading process stream for task: {}", taskId, e);
            TrainingTask t = taskManager.getTask(taskId);
            if (t != null) {
                synchronized (t) {
                    int currentSeq = t.getNextSequence();
                    TrainingLogEntry logEntry = TrainingLogEntry.error(
                        t.getTaskId(), currentSeq, "Stream read error: " + e.getMessage());
                    t.addLog(logEntry);
                    
                    TrainingLogEntryEntity logEntity = TrainingLogEntryEntity.fromTrainingLogEntry(
                        t.getTaskId(), logEntry);
                    addLogToQueue(taskId, logEntity);
                }
            }
            flushLogQueue(taskId);
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

    public String startEnhancedTraining(Fno2EnhancedTrainRequest request) {
        return startEnhancedTraining(request, defaultWorkingDir);
    }

    public String startEnhancedTraining(Fno2EnhancedTrainRequest request, String workingDir) {
        String taskId = taskManager.createTask("fno2-enhanced");
        TrainingTask task = taskManager.getTask(taskId);

        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Starting FNO2 Enhanced training task..."));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Working directory: " + workingDir));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "classNum: " + request.getClassNum()));

        try {
            List<String> command = buildEnhancedCommand(request);
            String commandStr = String.join(" ", command);
            task.setCommand(commandStr);
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Command: " + commandStr));
            
            createTaskEntity(taskId, commandStr, 
                TrainingTaskEntity.TaskType.TRAINING, "pytorch-fno2-enhanced");

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDir));

            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");
            
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Environment: PYTHONUNBUFFERED=1 (no buffering)"));

            task.markRunning();
            updateTaskStatus(taskId, TrainingTaskEntity.Status.RUNNING);

            Process process = processBuilder.start();
            runningProcesses.put(taskId, process);

            Future<?> future = executorService.submit(() -> {
                try {
                    runTrainingProcess(taskId, process);
                } catch (Exception e) {
                    log.error("[FnoTrainService] Error in enhanced training process for task: {}", taskId, e);
                    TrainingTask t = taskManager.getTask(taskId);
                    if (t != null) {
                        t.addLog(TrainingLogEntry.error(t.getTaskId(), t.getNextSequence(), "Error: " + e.getMessage()));
                        t.markFailed(e.getMessage());
                    }
                } finally {
                    runningProcesses.remove(taskId);
                    runningFutures.remove(taskId);
                    flushLogQueue(taskId);
                }
            });

            runningFutures.put(taskId, future);

            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Enhanced training started with task ID: " + taskId));
            return taskId;

        } catch (Exception e) {
            log.error("[FnoTrainService] Failed to start enhanced training task: {}", taskId, e);
            task.addLog(TrainingLogEntry.error(taskId, task.getNextSequence(), "Failed to start: " + e.getMessage()));
            task.markFailed(e.getMessage());
            updateTaskStatus(taskId, TrainingTaskEntity.Status.FAILED);
            return taskId;
        }
    }

    private List<String> buildEnhancedCommand(Fno2EnhancedTrainRequest request) {
        List<String> command = new ArrayList<>();

        command.add(config.getPythonExecutable());
        command.add("-u");

        command.add("train_fno2_enhanced.py");

        if (request.getDataRoot() != null && !request.getDataRoot().isEmpty()) {
            command.add("--data_root");
            command.add(request.getDataRoot());
        }

        if (request.getModelOutput() != null && !request.getModelOutput().isEmpty()) {
            command.add("--model_output");
            command.add(request.getModelOutput());
        }

        if (request.getLogFile() != null && !request.getLogFile().isEmpty()) {
            command.add("--log_file");
            command.add(request.getLogFile());
        }

        if (request.getProblems() != null && !request.getProblems().isEmpty()) {
            command.add("--problems");
            command.add(request.getProblems());
        }

        if (request.getCategories() != null && !request.getCategories().isEmpty()) {
            command.add("--categories");
            command.add(request.getCategories());
        }

        if (request.getClassNum() != null) {
            command.add("--classNum");
            command.add(request.getClassNum().toString());
        }

        if (request.getEpochs() != null) {
            command.add("--epochs");
            command.add(request.getEpochs().toString());
        }

        if (request.getBatchSize() != null) {
            command.add("--batch_size");
            command.add(request.getBatchSize().toString());
        }

        if (request.getLearningRate() != null) {
            command.add("--learning_rate");
            command.add(request.getLearningRate().toString());
        }

        if (request.getModes() != null) {
            command.add("--modes");
            command.add(request.getModes().toString());
        }

        if (request.getWidth() != null) {
            command.add("--width");
            command.add(request.getWidth().toString());
        }

        if (request.getMaxCasesPerCategory() != null) {
            command.add("--max_cases_per_category");
            command.add(request.getMaxCasesPerCategory().toString());
        }

        if (request.getTrainRatio() != null) {
            command.add("--train_ratio");
            command.add(request.getTrainRatio().toString());
        }

        if (request.getPrintEvery() != null) {
            command.add("--print_every");
            command.add(request.getPrintEvery().toString());
        }

        if (request.getDevice() != null && !request.getDevice().isEmpty()) {
            command.add("--device");
            command.add(request.getDevice());
        }

        if (request.getNoGradientStats() != null && request.getNoGradientStats()) {
            command.add("--no_gradient_stats");
        }

        if (request.getNoParamStats() != null && request.getNoParamStats()) {
            command.add("--no_param_stats");
        }

        if (request.getNoTensorStats() != null && request.getNoTensorStats()) {
            command.add("--no_tensor_stats");
        }

        if (request.getNoPredictionSamples() != null && request.getNoPredictionSamples()) {
            command.add("--no_prediction_samples");
        }

        if (request.getLogLevel() != null && !request.getLogLevel().isEmpty()) {
            command.add("--log_level");
            command.add(request.getLogLevel());
        }

        if (request.getQuiet() != null && request.getQuiet()) {
            command.add("--quiet");
        }

        if (request.getVerbose() != null && request.getVerbose()) {
            command.add("--verbose");
        }

        if (request.getNoFileLog() != null && request.getNoFileLog()) {
            command.add("--no_file_log");
        }

        return command;
    }

    public String quickTestEnhancedTraining() {
        Fno2EnhancedTrainRequest request = new Fno2EnhancedTrainRequest();
        request.setClassNum(1);
        request.setEpochs(1);
        request.setMaxCasesPerCategory(2);
        request.setBatchSize(4);
        request.setVerbose(true);
        return startEnhancedTraining(request);
    }

    public String quickTestEnhancedTrainingFull() {
        Fno2EnhancedTrainRequest request = new Fno2EnhancedTrainRequest();
        request.setClassNum(3);
        request.setEpochs(3);
        request.setMaxCasesPerCategory(5);
        request.setBatchSize(8);
        request.setLogLevel("DEBUG");
        return startEnhancedTraining(request);
    }

    public String startEvaluation(Fno2EvaluateRequest request) {
        return startEvaluation(request, defaultWorkingDir);
    }

    public String startEvaluation(Fno2EvaluateRequest request, String workingDir) {
        String taskId = taskManager.createTask("fno2-evaluate");
        TrainingTask task = taskManager.getTask(taskId);

        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Starting FNO2 Model Evaluation task..."));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Working directory: " + workingDir));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Model path: " + request.getModelPath()));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Metrics: " + request.getMetrics()));

        try {
            List<String> command = buildEvaluateCommand(request);
            String commandStr = String.join(" ", command);
            task.setCommand(commandStr);
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Command: " + commandStr));
            
            createTaskEntity(taskId, commandStr, 
                TrainingTaskEntity.TaskType.EVALUATION, "pytorch-evaluation");

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDir));

            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");
            
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Environment: PYTHONUNBUFFERED=1 (no buffering)"));

            task.markRunning();
            updateTaskStatus(taskId, TrainingTaskEntity.Status.RUNNING);

            Process process = processBuilder.start();
            runningProcesses.put(taskId, process);

            Future<?> future = executorService.submit(() -> {
                try {
                    runTrainingProcess(taskId, process);
                } catch (Exception e) {
                    log.error("[FnoTrainService] Error in evaluation process for task: {}", taskId, e);
                    TrainingTask t = taskManager.getTask(taskId);
                    if (t != null) {
                        t.addLog(TrainingLogEntry.error(t.getTaskId(), t.getNextSequence(), "Error: " + e.getMessage()));
                        t.markFailed(e.getMessage());
                    }
                } finally {
                    runningProcesses.remove(taskId);
                    runningFutures.remove(taskId);
                    flushLogQueue(taskId);
                }
            });

            runningFutures.put(taskId, future);

            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Evaluation started with task ID: " + taskId));
            return taskId;

        } catch (Exception e) {
            log.error("[FnoTrainService] Failed to start evaluation task: {}", taskId, e);
            task.addLog(TrainingLogEntry.error(taskId, task.getNextSequence(), "Failed to start: " + e.getMessage()));
            task.markFailed(e.getMessage());
            updateTaskStatus(taskId, TrainingTaskEntity.Status.FAILED);
            return taskId;
        }
    }

    private List<String> buildEvaluateCommand(Fno2EvaluateRequest request) {
        List<String> command = new ArrayList<>();

        command.add(config.getPythonExecutable());
        command.add("-u");

        command.add("evaluate_navier_stokes_enhanced.py");

        if (request.getModelPath() != null && !request.getModelPath().isEmpty()) {
            command.add("--model_path");
            command.add(request.getModelPath());
        }

        if (request.getDataRoot() != null && !request.getDataRoot().isEmpty()) {
            command.add("--data_root");
            command.add(request.getDataRoot());
        }

        if (request.getLogFile() != null && !request.getLogFile().isEmpty()) {
            command.add("--log_file");
            command.add(request.getLogFile());
        }

        if (request.getOutputReport() != null && !request.getOutputReport().isEmpty()) {
            command.add("--output_report");
            command.add(request.getOutputReport());
        }

        if (request.getProblems() != null && !request.getProblems().isEmpty()) {
            command.add("--problems");
            command.add(request.getProblems());
        }

        if (request.getCategories() != null && !request.getCategories().isEmpty()) {
            command.add("--categories");
            command.add(request.getCategories());
        }

        if (request.getMaxCasesPerCategory() != null) {
            command.add("--max_cases_per_category");
            command.add(request.getMaxCasesPerCategory().toString());
        }

        if (request.getBatchSize() != null) {
            command.add("--batch_size");
            command.add(request.getBatchSize().toString());
        }

        if (request.getPrintEvery() != null) {
            command.add("--print_every");
            command.add(request.getPrintEvery().toString());
        }

        if (request.getDevice() != null && !request.getDevice().isEmpty()) {
            command.add("--device");
            command.add(request.getDevice());
        }

        if (request.getMetrics() != null && !request.getMetrics().isEmpty()) {
            command.add("--metrics");
            command.add(request.getMetrics());
        }

        if (request.getLogLevel() != null && !request.getLogLevel().isEmpty()) {
            command.add("--log_level");
            command.add(request.getLogLevel());
        }

        if (request.getQuiet() != null && request.getQuiet()) {
            command.add("--quiet");
        }

        if (request.getVerbose() != null && request.getVerbose()) {
            command.add("--verbose");
        }

        if (request.getNoFileLog() != null && request.getNoFileLog()) {
            command.add("--no_file_log");
        }

        if (request.getListMetrics() != null && request.getListMetrics()) {
            command.add("--list_metrics");
        }

        return command;
    }

    public String quickTestEvaluation() {
        Fno2EvaluateRequest request = new Fno2EvaluateRequest();
        request.setModelPath("fno2_full_model.pth");
        request.setMaxCasesPerCategory(2);
        request.setMetrics("all");
        request.setVerbose(true);
        return startEvaluation(request);
    }

    public String quickTestEvaluationBasic() {
        Fno2EvaluateRequest request = new Fno2EvaluateRequest();
        request.setModelPath("fno2_full_model.pth");
        request.setMaxCasesPerCategory(2);
        request.setMetrics("mse,rmse,r2");
        return startEvaluation(request);
    }

    public String startPaddleTraining(Fno2PaddleTrainRequest request) {
        return startPaddleTraining(request, defaultWorkingDir);
    }

    public String startPaddleTraining(Fno2PaddleTrainRequest request, String workingDir) {
        String taskId = taskManager.createTask("fno2-paddle");
        TrainingTask task = taskManager.getTask(taskId);

        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Starting FNO2 PaddlePaddle training task..."));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Framework: PaddlePaddle"));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Working directory: " + workingDir));
        task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "classNum: " + request.getClassNum()));

        try {
            List<String> command = buildPaddleCommand(request);
            String commandStr = String.join(" ", command);
            task.setCommand(commandStr);
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Command: " + commandStr));
            
            createTaskEntity(taskId, commandStr, 
                TrainingTaskEntity.TaskType.PADDLE_TRAINING, "paddlepaddle-fno2");

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDir));

            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");
            
            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "Environment: PYTHONUNBUFFERED=1 (no buffering)"));

            task.markRunning();
            updateTaskStatus(taskId, TrainingTaskEntity.Status.RUNNING);

            Process process = processBuilder.start();
            runningProcesses.put(taskId, process);

            Future<?> future = executorService.submit(() -> {
                try {
                    runTrainingProcess(taskId, process);
                } catch (Exception e) {
                    log.error("[FnoTrainService] Error in PaddlePaddle training process for task: {}", taskId, e);
                    TrainingTask t = taskManager.getTask(taskId);
                    if (t != null) {
                        t.addLog(TrainingLogEntry.error(t.getTaskId(), t.getNextSequence(), "Error: " + e.getMessage()));
                        t.markFailed(e.getMessage());
                    }
                } finally {
                    runningProcesses.remove(taskId);
                    runningFutures.remove(taskId);
                    flushLogQueue(taskId);
                }
            });

            runningFutures.put(taskId, future);

            task.addLog(TrainingLogEntry.info(taskId, task.getNextSequence(), "PaddlePaddle training started with task ID: " + taskId));
            return taskId;

        } catch (Exception e) {
            log.error("[FnoTrainService] Failed to start PaddlePaddle training task: {}", taskId, e);
            task.addLog(TrainingLogEntry.error(taskId, task.getNextSequence(), "Failed to start: " + e.getMessage()));
            task.markFailed(e.getMessage());
            updateTaskStatus(taskId, TrainingTaskEntity.Status.FAILED);
            return taskId;
        }
    }

    private List<String> buildPaddleCommand(Fno2PaddleTrainRequest request) {
        List<String> command = new ArrayList<>();

        command.add(config.getPythonExecutable());
        command.add("-u");

        command.add("train_fno2_paddle.py");

        if (request.getDataRoot() != null && !request.getDataRoot().isEmpty()) {
            command.add("--data_root");
            command.add(request.getDataRoot());
        }

        if (request.getModelOutput() != null && !request.getModelOutput().isEmpty()) {
            command.add("--model_output");
            command.add(request.getModelOutput());
        }

        if (request.getLogFile() != null && !request.getLogFile().isEmpty()) {
            command.add("--log_file");
            command.add(request.getLogFile());
        }

        if (request.getProblems() != null && !request.getProblems().isEmpty()) {
            command.add("--problems");
            command.add(request.getProblems());
        }

        if (request.getCategories() != null && !request.getCategories().isEmpty()) {
            command.add("--categories");
            command.add(request.getCategories());
        }

        if (request.getClassNum() != null) {
            command.add("--classNum");
            command.add(request.getClassNum().toString());
        }

        if (request.getEpochs() != null) {
            command.add("--epochs");
            command.add(request.getEpochs().toString());
        }

        if (request.getBatchSize() != null) {
            command.add("--batch_size");
            command.add(request.getBatchSize().toString());
        }

        if (request.getLearningRate() != null) {
            command.add("--learning_rate");
            command.add(request.getLearningRate().toString());
        }

        if (request.getModes() != null) {
            command.add("--modes");
            command.add(request.getModes().toString());
        }

        if (request.getWidth() != null) {
            command.add("--width");
            command.add(request.getWidth().toString());
        }

        if (request.getMaxCasesPerCategory() != null) {
            command.add("--max_cases_per_category");
            command.add(request.getMaxCasesPerCategory().toString());
        }

        if (request.getTrainRatio() != null) {
            command.add("--train_ratio");
            command.add(request.getTrainRatio().toString());
        }

        if (request.getPrintEvery() != null) {
            command.add("--print_every");
            command.add(request.getPrintEvery().toString());
        }

        if (request.getDevice() != null && !request.getDevice().isEmpty()) {
            command.add("--device");
            command.add(request.getDevice());
        }

        if (request.getNoGradientStats() != null && request.getNoGradientStats()) {
            command.add("--no_gradient_stats");
        }

        if (request.getNoParamStats() != null && request.getNoParamStats()) {
            command.add("--no_param_stats");
        }

        if (request.getNoTensorStats() != null && request.getNoTensorStats()) {
            command.add("--no_tensor_stats");
        }

        if (request.getNoPredictionSamples() != null && request.getNoPredictionSamples()) {
            command.add("--no_prediction_samples");
        }

        if (request.getLogLevel() != null && !request.getLogLevel().isEmpty()) {
            command.add("--log_level");
            command.add(request.getLogLevel());
        }

        if (request.getQuiet() != null && request.getQuiet()) {
            command.add("--quiet");
        }

        if (request.getVerbose() != null && request.getVerbose()) {
            command.add("--verbose");
        }

        if (request.getNoFileLog() != null && request.getNoFileLog()) {
            command.add("--no_file_log");
        }

        return command;
    }

    public String quickTestPaddleTraining() {
        Fno2PaddleTrainRequest request = new Fno2PaddleTrainRequest();
        request.setClassNum(1);
        request.setEpochs(1);
        request.setMaxCasesPerCategory(2);
        request.setBatchSize(4);
        request.setVerbose(true);
        return startPaddleTraining(request);
    }

    public String quickTestPaddleTrainingFull() {
        Fno2PaddleTrainRequest request = new Fno2PaddleTrainRequest();
        request.setClassNum(3);
        request.setEpochs(3);
        request.setMaxCasesPerCategory(5);
        request.setBatchSize(8);
        request.setLogLevel("DEBUG");
        return startPaddleTraining(request);
    }
}
