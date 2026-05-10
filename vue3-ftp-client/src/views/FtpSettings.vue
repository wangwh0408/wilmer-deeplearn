<template>
  <div class="settings-container">
    <el-card class="settings-card">
      <template #header>
        <div class="card-header">
          <el-icon><Setting /></el-icon>
          <span>FTP服务器设置</span>
        </div>
      </template>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-width="120px"
        class="settings-form"
      >
        <el-form-item label="服务器名称" prop="serverName">
          <el-input v-model="form.serverName" placeholder="例如: default, production" />
          <el-text type="info" size="small">可选，用于标识不同的FTP服务器</el-text>
        </el-form-item>

        <el-divider content-position="left">连接配置</el-divider>

        <el-row :gutter="20">
          <el-col :span="18">
            <el-form-item label="服务器地址" prop="host">
              <el-input v-model="form.host" placeholder="例如: ftp.example.com 或 192.168.1.100" />
            </el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="端口" prop="port">
              <el-input-number
                v-model="form.port"
                :min="1"
                :max="65535"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="用户名" prop="username">
              <el-input v-model="form.username" placeholder="用户名" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="密码" prop="password">
              <el-input
                v-model="form.password"
                type="password"
                placeholder="密码"
                show-password
              />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="根目录">
          <el-input v-model="form.rootPath" placeholder="例如: /home/user 或 /" />
          <el-text type="info" size="small">连接后默认进入的目录</el-text>
        </el-form-item>

        <el-divider content-position="left">高级配置</el-divider>

        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="连接超时">
              <el-input-number
                v-model="form.connectTimeout"
                :min="1000"
                :max="300000"
                :step="1000"
                style="width: 100%"
              />
              <el-text type="info" size="small">毫秒</el-text>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="数据超时">
              <el-input-number
                v-model="form.dataTimeout"
                :min="1000"
                :max="600000"
                :step="5000"
                style="width: 100%"
              />
              <el-text type="info" size="small">毫秒</el-text>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="编码">
              <el-select v-model="form.encoding" style="width: 100%">
                <el-option label="UTF-8" value="UTF-8" />
                <el-option label="GBK" value="GBK" />
                <el-option label="GB2312" value="GB2312" />
                <el-option label="ISO-8859-1" value="ISO-8859-1" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="被动模式">
          <el-switch v-model="form.passiveMode" />
          <el-text type="info" size="small" style="margin-left: 10px">
            大多数情况下建议启用被动模式
          </el-text>
        </el-form-item>

        <el-divider />

        <el-form-item>
          <el-space>
            <el-button type="primary" @click="testConnection" :loading="testing">
              <el-icon><Connection /></el-icon>
              测试连接
            </el-button>
            <el-button type="success" @click="saveAndConnect">
              <el-icon><Check /></el-icon>
              保存并连接
            </el-button>
            <el-button @click="loadSavedConfig">
              <el-icon><Refresh /></el-icon>
              恢复默认
            </el-button>
          </el-space>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="saved-servers-card">
      <template #header>
        <div class="card-header">
          <el-icon><Collection /></el-icon>
          <span>已保存的服务器</span>
        </div>
      </template>

      <el-table :data="savedServers" style="width: 100%">
        <el-table-column prop="name" label="名称" width="150" />
        <el-table-column prop="host" label="地址" />
        <el-table-column prop="port" label="端口" width="80" />
        <el-table-column prop="username" label="用户名" width="120" />
        <el-table-column label="操作" width="180">
          <template #default="scope">
            <el-button size="small" type="primary" @click="connectToServer(scope.row)">
              连接
            </el-button>
            <el-button size="small" type="danger" @click="deleteServer(scope.$index)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ftpApi } from '@/api/ftp'

const router = useRouter()
const formRef = ref(null)
const testing = ref(false)
const savedServers = ref([])

const form = reactive({
  serverName: '',
  host: '',
  port: 21,
  username: 'anonymous',
  password: '',
  rootPath: '/',
  connectTimeout: 30000,
  dataTimeout: 300000,
  passiveMode: true,
  encoding: 'UTF-8'
})

const rules = {
  host: [{ required: true, message: '请输入服务器地址', trigger: 'blur' }]
}

const loadSavedConfig = () => {
  const saved = localStorage.getItem('ftp_connections')
  if (saved) {
    try {
      savedServers.value = JSON.parse(saved)
    } catch (e) {
      savedServers.value = []
    }
  }
  
  const last = localStorage.getItem('ftp_last_connection')
  if (last) {
    try {
      const lastConfig = JSON.parse(last)
      Object.assign(form, lastConfig)
    } catch (e) {
      console.error('Failed to load last config:', e)
    }
  }
}

const saveConnection = () => {
  localStorage.setItem('ftp_last_connection', JSON.stringify(form))
  ElMessage.success('配置已保存')
}

const connectToServer = (server) => {
  Object.assign(form, {
    serverName: server.name,
    host: server.host,
    port: server.port,
    username: server.username,
    password: server.password || '',
    rootPath: server.rootPath || '/',
    connectTimeout: server.connectTimeout || 30000,
    dataTimeout: server.dataTimeout || 300000,
    passiveMode: server.passiveMode !== false,
    encoding: server.encoding || 'UTF-8'
  })
  saveAndConnect()
}

const deleteServer = (index) => {
  ElMessageBox.confirm('确定要删除这个服务器配置吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    savedServers.value.splice(index, 1)
    localStorage.setItem('ftp_connections', JSON.stringify(savedServers.value))
    ElMessage.success('已删除')
  }).catch(() => {})
}

const testConnection = async () => {
  if (!form.host) {
    ElMessage.warning('请输入服务器地址')
    return
  }

  testing.value = true
  try {
    const result = await ftpApi.testConnection({
      host: form.host,
      port: form.port,
      username: form.username,
      password: form.password,
      rootPath: form.rootPath,
      connectTimeout: form.connectTimeout,
      dataTimeout: form.dataTimeout,
      passiveMode: form.passiveMode,
      encoding: form.encoding
    })

    if (result.data.connected) {
      ElMessage.success('连接成功！')
    } else {
      ElMessage.error('连接失败')
    }
  } catch (error) {
    console.error('Test connection failed:', error)
  } finally {
    testing.value = false
  }
}

const saveAndConnect = () => {
  if (!form.host) {
    ElMessage.warning('请输入服务器地址')
    return
  }

  saveConnection()
  
  sessionStorage.setItem('ftp_current_connection', JSON.stringify(form))
  ElMessage.success('已连接，正在跳转到文件浏览器...')
  
  setTimeout(() => {
    router.push('/ftp')
  }, 500)
}

onMounted(() => {
  loadSavedConfig()
})
</script>

<style scoped>
.settings-container {
  max-width: 900px;
  margin: 0 auto;
}

.settings-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
}

.settings-form {
  max-width: 800px;
}

.saved-servers-card {
  margin-top: 20px;
}
</style>
