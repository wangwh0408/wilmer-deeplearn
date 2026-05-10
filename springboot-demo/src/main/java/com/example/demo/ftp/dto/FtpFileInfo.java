package com.example.demo.ftp.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FtpFileInfo implements Serializable {

    private String name;

    private String path;

    private String absolutePath;

    private String parentPath;

    @JsonProperty("isDirectory")
    private boolean isDirectory;

    @JsonProperty("isFile")
    private boolean isFile;

    @JsonProperty("isSymbolicLink")
    private boolean isSymbolicLink;

    private long size;

    private String formattedSize;

    private LocalDateTime lastModified;

    private String lastModifiedStr;

    private String permissions;

    private String owner;

    private String group;

    private int hardLinkCount;

    public String getFormattedSize() {
        if (formattedSize != null) {
            return formattedSize;
        }
        return formatSize(size);
    }

    private String formatSize(long size) {
        if (size < 1024) {
            return size + " B";
        } else if (size < 1024 * 1024) {
            return String.format("%.2f KB", size / 1024.0);
        } else if (size < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", size / (1024.0 * 1024));
        } else {
            return String.format("%.2f GB", size / (1024.0 * 1024 * 1024));
        }
    }
}
