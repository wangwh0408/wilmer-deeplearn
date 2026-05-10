<template>
  <el-config-provider :locale="zhCn">
    <el-container class="app-container">
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo">
            <el-icon :size="28"><FolderOpened /></el-icon>
            <span class="title">FTP Client</span>
          </div>
          <el-menu 
            :default-active="activeMenu" 
            mode="horizontal" 
            background-color="transparent"
            text-color="#fff"
            active-text-color="#ffd04b"
            router
          >
            <el-menu-item index="/ftp">
              <el-icon><FolderOpened /></el-icon>
              <span>文件浏览器</span>
            </el-menu-item>
            <el-menu-item index="/settings">
              <el-icon><Setting /></el-icon>
              <span>设置</span>
            </el-menu-item>
          </el-menu>
        </div>
      </el-header>
      
      <el-main class="app-main">
        <router-view />
      </el-main>
      
      <el-footer class="app-footer">
        <span>Vue3 FTP Client v1.0.0 | 高可用FTP服务</span>
      </el-footer>
    </el-container>
  </el-config-provider>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

const route = useRoute()

const activeMenu = computed(() => route.path)
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  padding: 0;
  height: 60px;
  line-height: 60px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  max-width: 1600px;
  margin: 0 auto;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #fff;
}

.title {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 1px;
}

.app-main {
  flex: 1;
  padding: 20px;
  background: #f5f7fa;
  overflow: auto;
}

.app-footer {
  height: 40px;
  line-height: 40px;
  text-align: center;
  background: #303133;
  color: #909399;
  font-size: 13px;
}

:deep(.el-menu--horizontal.el-menu) {
  border-bottom: none;
  background: transparent;
}

:deep(.el-menu--horizontal > .el-menu-item) {
  border-bottom: none;
}

:deep(.el-menu--horizontal > .el-menu-item:hover),
:deep(.el-menu--horizontal > .el-menu-item:focus) {
  background-color: rgba(255, 255, 255, 0.1);
}
</style>
