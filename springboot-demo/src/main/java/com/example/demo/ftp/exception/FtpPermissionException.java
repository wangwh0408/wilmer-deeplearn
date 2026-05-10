package com.example.demo.ftp.exception;

public class FtpPermissionException extends FtpException {

    public FtpPermissionException(String operation, String path) {
        super(403, "Permission denied: " + operation + " on " + path);
    }

    public FtpPermissionException(String operation, String path, String ftpReplyCode) {
        super(403, "Permission denied: " + operation + " on " + path, ftpReplyCode);
    }

    public FtpPermissionException(String message) {
        super(403, "Permission denied: " + message);
    }
}
