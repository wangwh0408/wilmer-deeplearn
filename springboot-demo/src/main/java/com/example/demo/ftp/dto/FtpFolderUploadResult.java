package com.example.demo.ftp.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FtpFolderUploadResult implements Serializable {

    private boolean success;

    private String folderName;

    private String targetBasePath;

    private int totalFiles;

    private int uploadedFiles;

    private int failedFiles;

    private List<FileUploadResult> results;

    private List<String> errorMessages;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FileUploadResult implements Serializable {
        private String relativePath;
        private String targetPath;
        private boolean success;
        private String errorMessage;
        private long fileSize;
    }

    public static FtpFolderUploadResult create(String folderName, String targetBasePath) {
        return FtpFolderUploadResult.builder()
            .folderName(folderName)
            .targetBasePath(targetBasePath)
            .totalFiles(0)
            .uploadedFiles(0)
            .failedFiles(0)
            .results(new ArrayList<>())
            .errorMessages(new ArrayList<>())
            .build();
    }

    public void addSuccess(String relativePath, String targetPath, long fileSize) {
        results.add(FileUploadResult.builder()
            .relativePath(relativePath)
            .targetPath(targetPath)
            .success(true)
            .fileSize(fileSize)
            .build());
        uploadedFiles++;
        totalFiles++;
    }

    public void addFailure(String relativePath, String targetPath, String errorMessage) {
        results.add(FileUploadResult.builder()
            .relativePath(relativePath)
            .targetPath(targetPath)
            .success(false)
            .errorMessage(errorMessage)
            .build());
        failedFiles++;
        totalFiles++;
        errorMessages.add("Failed to upload " + relativePath + ": " + errorMessage);
    }

    public boolean isAllSuccess() {
        return failedFiles == 0;
    }
}
