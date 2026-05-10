package com.example.demo.ftp.exception;

import com.example.demo.common.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.io.IOException;

@Slf4j
@RestControllerAdvice(basePackages = "com.example.demo.ftp")
public class FtpExceptionHandler {

    @ExceptionHandler(FtpException.class)
    public ResponseEntity<Result<?>> handleFtpException(FtpException e) {
        log.error("[FTP Exception] {}", e.getMessage(), e);
        
        Result<?> result = Result.error(e.getErrorCode(), e.getErrorMessage());
        return ResponseEntity.status(HttpStatus.OK).body(result);
    }

    @ExceptionHandler(FtpConnectionException.class)
    public ResponseEntity<Result<?>> handleFtpConnectionException(FtpConnectionException e) {
        log.error("[FTP Connection Exception] {}", e.getMessage(), e);
        
        Result<?> result = Result.error(e.getErrorCode(), e.getErrorMessage());
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(result);
    }

    @ExceptionHandler(FtpFileNotFoundException.class)
    public ResponseEntity<Result<?>> handleFtpFileNotFoundException(FtpFileNotFoundException e) {
        log.warn("[FTP File Not Found] {}", e.getMessage());
        
        Result<?> result = Result.error(e.getErrorCode(), e.getErrorMessage());
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(result);
    }

    @ExceptionHandler(FtpPermissionException.class)
    public ResponseEntity<Result<?>> handleFtpPermissionException(FtpPermissionException e) {
        log.warn("[FTP Permission Denied] {}", e.getMessage());
        
        Result<?> result = Result.error(e.getErrorCode(), e.getErrorMessage());
        return ResponseEntity.status(HttpStatus.FORBIDDEN).body(result);
    }

    @ExceptionHandler(FtpTimeoutException.class)
    public ResponseEntity<Result<?>> handleFtpTimeoutException(FtpTimeoutException e) {
        log.warn("[FTP Timeout] {}", e.getMessage());
        
        Result<?> result = Result.error(e.getErrorCode(), e.getErrorMessage());
        return ResponseEntity.status(HttpStatus.REQUEST_TIMEOUT).body(result);
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<Result<?>> handleMaxUploadSizeExceededException(MaxUploadSizeExceededException e) {
        log.warn("[Upload Size Exceeded] {}", e.getMessage());
        
        Result<?> result = Result.error(413, "File size exceeds maximum allowed limit");
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE).body(result);
    }

    @ExceptionHandler(IOException.class)
    public ResponseEntity<Result<?>> handleIOException(IOException e) {
        log.error("[IO Exception] {}", e.getMessage(), e);
        
        Result<?> result = Result.error(500, "IO Error: " + e.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(result);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Result<?>> handleIllegalArgumentException(IllegalArgumentException e) {
        log.warn("[Illegal Argument] {}", e.getMessage());
        
        Result<?> result = Result.error(400, "Invalid parameter: " + e.getMessage());
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(result);
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Result<?>> handleIllegalStateException(IllegalStateException e) {
        log.warn("[Illegal State] {}", e.getMessage());
        
        Result<?> result = Result.error(500, "Invalid state: " + e.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(result);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Result<?>> handleGenericException(Exception e) {
        log.error("[Unexpected Exception] {}", e.getMessage(), e);
        
        Result<?> result = Result.error(500, "Unexpected error: " + e.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(result);
    }
}
