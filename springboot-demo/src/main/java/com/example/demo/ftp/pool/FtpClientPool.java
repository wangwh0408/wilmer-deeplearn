package com.example.demo.ftp.pool;

import com.example.demo.ftp.config.FtpProperties;
import com.example.demo.ftp.dto.FtpConnectionRequest;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.net.ftp.FTPClient;
import org.apache.commons.pool2.impl.GenericKeyedObjectPool;
import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;

@Slf4j
public class FtpClientPool {

    private final GenericKeyedObjectPool<FtpConnectionRequest, FTPClient> pool;
    private final FtpProperties properties;
    private final ConcurrentHashMap<FtpConnectionRequest, FtpConnectionRequest> requestCache = new ConcurrentHashMap<>();
    private final Lock lock = new ReentrantLock();

    public FtpClientPool(FtpClientFactory factory, FtpProperties properties) {
        this.properties = properties;
        
        GenericKeyedObjectPoolConfig<FTPClient> config = new GenericKeyedObjectPoolConfig<>();
        FtpProperties.Pool poolConfig = properties.getPool();
        
        config.setMaxTotalPerKey(poolConfig.getMaxTotal());
        config.setMaxIdlePerKey(poolConfig.getMaxIdle());
        config.setMinIdlePerKey(poolConfig.getMinIdle());
        config.setMaxWaitMillis(poolConfig.getMaxWaitMillis());
        config.setTestOnBorrow(poolConfig.isTestOnBorrow());
        config.setTestOnReturn(poolConfig.isTestOnReturn());
        config.setTestWhileIdle(poolConfig.isTestWhileIdle());
        config.setTimeBetweenEvictionRunsMillis(poolConfig.getTimeBetweenEvictionRunsMillis());
        config.setMinEvictableIdleTimeMillis(poolConfig.getMinEvictableIdleTimeMillis());
        config.setSoftMinEvictableIdleTimeMillis(poolConfig.getSoftMinEvictableIdleTimeMillis());
        config.setNumTestsPerEvictionRun(poolConfig.getNumTestsPerEvictionRun());
        config.setTestOnCreate(poolConfig.isTestOnCreate());
        config.setJmxEnabled(true);
        config.setJmxNamePrefix("ftp-client-pool");
        config.setBlockWhenExhausted(true);
        
        this.pool = new GenericKeyedObjectPool<>(factory, config);
        
        log.info("[FtpClientPool] FTP Client Pool initialized with config: maxTotal={}, maxIdle={}, minIdle={}",
            poolConfig.getMaxTotal(), poolConfig.getMaxIdle(), poolConfig.getMinIdle());
    }

    public FTPClient borrowObject(FtpConnectionRequest request) throws Exception {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        
        FtpConnectionRequest cachedRequest = getCachedRequest(request);
        log.debug("[FtpClientPool] Borrowing FTP client for {}:{}", cachedRequest.getHost(), cachedRequest.getPort());
        
        FTPClient client = pool.borrowObject(cachedRequest);
        
        if (log.isDebugEnabled()) {
            logPoolStats("After borrow");
        }
        
        return client;
    }

    public void returnObject(FtpConnectionRequest request, FTPClient client) {
        if (client == null) {
            log.warn("[FtpClientPool] Cannot return null client");
            return;
        }
        
        if (request == null) {
            log.warn("[FtpClientPool] Cannot return with null request");
            return;
        }
        
        FtpConnectionRequest cachedRequest = getCachedRequest(request);
        
        try {
            log.debug("[FtpClientPool] Returning FTP client for {}:{}", cachedRequest.getHost(), cachedRequest.getPort());
            pool.returnObject(cachedRequest, client);
            
            if (log.isDebugEnabled()) {
                logPoolStats("After return");
            }
        } catch (Exception e) {
            log.error("[FtpClientPool] Error returning object to pool: {}", e.getMessage(), e);
            invalidateObject(cachedRequest, client);
        }
    }

    public void invalidateObject(FtpConnectionRequest request, FTPClient client) {
        if (client == null) {
            log.warn("[FtpClientPool] Cannot invalidate null client");
            return;
        }
        
        if (request == null) {
            log.warn("[FtpClientPool] Cannot invalidate with null request");
            return;
        }
        
        FtpConnectionRequest cachedRequest = getCachedRequest(request);
        
        try {
            log.warn("[FtpClientPool] Invalidating FTP client for {}:{}", cachedRequest.getHost(), cachedRequest.getPort());
            pool.invalidateObject(cachedRequest, client);
            
            if (log.isDebugEnabled()) {
                logPoolStats("After invalidate");
            }
        } catch (Exception e) {
            log.error("[FtpClientPool] Error invalidating object: {}", e.getMessage(), e);
        }
    }

    public void clear(FtpConnectionRequest request) {
        if (request == null) {
            log.warn("[FtpClientPool] Cannot clear pool with null request");
            return;
        }
        
        FtpConnectionRequest cachedRequest = getCachedRequest(request);
        log.info("[FtpClientPool] Clearing pool for {}:{}", cachedRequest.getHost(), cachedRequest.getPort());
        pool.clear(cachedRequest);
    }

    public void clearAll() {
        log.info("[FtpClientPool] Clearing all pools");
        pool.clear();
        requestCache.clear();
    }

    public <T> T executeWithClient(FtpConnectionRequest request, Function<FTPClient, T> function) throws Exception {
        FTPClient client = null;
        boolean success = false;
        
        try {
            client = borrowObject(request);
            T result = function.apply(client);
            success = true;
            return result;
        } finally {
            if (client != null) {
                if (success) {
                    returnObject(request, client);
                } else {
                    invalidateObject(request, client);
                }
            }
        }
    }

    public int getNumActive(FtpConnectionRequest request) {
        FtpConnectionRequest cachedRequest = getCachedRequest(request);
        return pool.getNumActive(cachedRequest);
    }

    public int getNumIdle(FtpConnectionRequest request) {
        FtpConnectionRequest cachedRequest = getCachedRequest(request);
        return pool.getNumIdle(cachedRequest);
    }

    public int getNumActiveTotal() {
        return pool.getNumActive();
    }

    public int getNumIdleTotal() {
        return pool.getNumIdle();
    }

    private FtpConnectionRequest getCachedRequest(FtpConnectionRequest request) {
        return requestCache.computeIfAbsent(request, k -> k);
    }

    private void logPoolStats(String prefix) {
        log.debug("[FtpClientPool] {} - Active: {}, Idle: {}", 
            prefix, getNumActiveTotal(), getNumIdleTotal());
    }

    @PreDestroy
    public void shutdown() {
        log.info("[FtpClientPool] Shutting down FTP client pool...");
        
        try {
            pool.close();
            log.info("[FtpClientPool] FTP client pool closed successfully");
        } catch (Exception e) {
            log.error("[FtpClientPool] Error closing pool: {}", e.getMessage(), e);
        }
        
        requestCache.clear();
    }
}
