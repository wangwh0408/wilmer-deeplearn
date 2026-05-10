package com.example.demo.ftp.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FtpDirectoryListing {

    private String currentPath;

    private String parentPath;

    private boolean isRoot;

    private List<String> pathParts;

    private List<FtpFileInfo> files;

    private int totalFiles;

    private int directoryCount;

    private int fileCount;
}
