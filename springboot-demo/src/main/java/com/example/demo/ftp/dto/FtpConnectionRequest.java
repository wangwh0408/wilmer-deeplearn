package com.example.demo.ftp.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import javax.validation.constraints.NotBlank;
import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FtpConnectionRequest implements Serializable {

    private String serverName;

    @NotBlank(message = "Host cannot be blank")
    private String host;

    private int port = 21;

    private String username = "anonymous";

    private String password = "";

    private String rootPath = "/";

    private int connectTimeout = 30000;

    private int dataTimeout = 300000;

    private int defaultSoTimeout = 60000;

    private boolean passiveMode = true;

    private String encoding = "UTF-8";
}
