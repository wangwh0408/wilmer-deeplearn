package com.example.demo.ftp.exception;

public class FtpFileNotFoundException extends FtpException {

    public FtpFileNotFoundException(String path) {
        super(404, "File or directory not found: " + path);
    }

    public FtpFileNotFoundException(String path, String ftpReplyCode) {
        super(404, "File or directory not found: " + path, ftpReplyCode);
    }

    public FtpFileNotFoundException(String path, Throwable cause) {
        super(404, "File or directory not found: " + path, null, cause);
    }
}
