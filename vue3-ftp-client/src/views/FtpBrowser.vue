<template>
  <div class="ftp-browser-container">
    <el-card class="browser-card">
      <template #header>
        <div class="card-header">
          <div class="header-left">
            <el-icon><FolderOpened /></el-icon>
            <span>FTP文件浏览器</span>
            <el-tag v-if="connected" type="success" effect="dark">已连接</el-tag>
            <el-tag v-else type="danger" effect="dark">未连接</el-tag>
          </div>
          <div class="header-right">
            <el-space>
              <el-button type="primary" @click="goToSettings">
                <el-icon><Setting /></el-icon>
                连接设置
              </el-button>
              <el-button @click="refresh">
                <el-icon><Refresh /></el-icon>
                刷新
              </el-button>
              <el-button type="primary" @click="showUploadDialog = true">
                <el-icon><Upload /></el-icon>
                上传
              </el-button>
              <el-button type="success" @click="createNewFolder">
                <el-icon><FolderAdd /></el-icon>
                新建文件夹
              </el-button>
            </el-space>
          </div>
        </div>
      </template>

      <div v-if="!connected" class="not-connected">
        <el-empty description="未连接到FTP服务器">
          <el-button type="primary" @click="goToSettings">前往连接设置</el-button>
        </el-empty>
      </div>

      <div v-else>
        <el-breadcrumb separator="/" class="path-navigator">
          <el-breadcrumb-item v-for="(item, index) in pathParts" :key="index">
            <el-button type="text" @click="navigateTo(index)" class="path-btn">
              <el-icon v-if="index === 0"><HomeFilled /></el-icon>
              <span>{{ item }}</span>
            </el-button>
          </el-breadcrumb-item>
        </el-breadcrumb>

        <el-table
          :data="fileList"
          v-loading="loading"
          @row-dblclick="handleRowDblclick"
          stripe
          style="width: 100%"
          max-height="600"
        >
          <el-table-column type="selection" width="50" />
          
          <el-table-column prop="name" label="名称" min-width="200">
            <template #default="scope">
              <div class="file-name-cell" @click="handleNameClick(scope.row)">
                <el-icon v-if="scope.row.isDirectory" class="dir-icon" :size="20">
                  <Folder />
                </el-icon>
                <el-icon v-else-if="isImage(scope.row.name)" class="file-icon" :size="20">
                  <Picture />
                </el-icon>
                <el-icon v-else-if="isZip(scope.row.name)" class="file-icon" :size="20">
                  <ZipFolder />
                </el-icon>
                <el-icon v-else-if="isDocument(scope.row.name)" class="file-icon" :size="20">
                  <Document />
                </el-icon>
                <el-icon v-else class="file-icon" :size="20">
                  <Files />
                </el-icon>
                <span class="name-text">{{ scope.row.name }}</span>
              </div>
            </template>
          </el-table-column>

          <el-table-column prop="formattedSize" label="大小" width="120" align="right">
            <template #default="scope">
              <span v-if="!scope.row.isDirectory">{{ scope.row.formattedSize }}</span>
              <span v-else class="dir-size">-</span>
            </template>
          </el-table-column>

          <el-table-column prop="lastModifiedStr" label="修改时间" width="180" />
          
          <el-table-column prop="permissions" label="权限" width="100" />
          
          <el-table-column label="操作" width="200" fixed="right">
            <template #default="scope">
              <el-space>
                <el-button
                  v-if="scope.row.isDirectory"
                  size="small"
                  type="primary"
                  @click="enterDirectory(scope.row)"
                >
                  进入
                </el-button>
                <el-button
                  v-else
                  size="small"
                  type="primary"
                  @click="downloadFile(scope.row)"
                >
                  下载
                </el-button>
                <el-button size="small" type="warning" @click="openRenameDialog(scope.row)">
                  重命名
                </el-button>
                <el-button size="small" type="danger" @click="handleDelete(scope.row)">
                  删除
                </el-button>
              </el-space>
            </template>
          </el-table-column>
        </el-table>

        <div class="pagination-info">
          <el-text>共 {{ totalFiles }} 个项目，其中 {{ directoryCount }} 个文件夹，{{ fileCount }} 个文件</el-text>
        </div>
      </div>
    </el-card>

    <el-dialog v-model="showUploadDialog" title="上传文件" width="650px">
      <el-tabs v-model="uploadMode" type="card">
        <el-tab-pane label="上传文件" name="files">
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :file-list="uploadFileList"
            :on-change="handleUploadChange"
            :on-remove="handleUploadRemove"
            multiple
            drag
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              将文件拖到此处，或<em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">支持上传单个或多个文件</div>
            </template>
          </el-upload>
        </el-tab-pane>
        
        <el-tab-pane label="上传文件夹" name="folder">
          <div class="folder-upload-area" @click="triggerFolderSelect">
            <el-icon class="el-icon--upload" :size="40"><FolderAdd /></el-icon>
            <div class="folder-upload-text">
              点击选择文件夹
            </div>
            <input
              type="file"
              ref="folderInputRef"
              style="display: none"
              webkitdirectory
              multiple
              @change="handleFolderSelect"
            />
          </div>
          
          <div v-if="folderFileList.length > 0" class="folder-preview">
            <el-divider content-position="left">待上传文件 ({{ folderFileList.length }} 个)</el-divider>
            <div class="folder-file-list">
              <div v-for="(file, index) in folderFileList.slice(0, 20)" :key="index" class="folder-file-item">
                <el-icon><Files /></el-icon>
                <span class="folder-file-name">{{ file.relativePath }}</span>
                <span class="folder-file-size">{{ formatFileSize(file.raw.size) }}</span>
              </div>
              <div v-if="folderFileList.length > 20" class="more-files">
                ...还有 {{ folderFileList.length - 20 }} 个文件
              </div>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
      
      <el-checkbox v-model="overwriteUpload" style="margin-top: 15px">
        覆盖已存在的文件
      </el-checkbox>

      <template #footer>
        <el-space>
          <el-button @click="closeUploadDialog">取消</el-button>
          <el-button 
            type="primary" 
            @click="startUpload" 
            :loading="uploading"
            :disabled="(uploadMode === 'files' && uploadFileList.length === 0) || (uploadMode === 'folder' && folderFileList.length === 0)"
          >
            开始上传
          </el-button>
        </el-space>
      </template>
    </el-dialog>

    <el-dialog v-model="showNewFolderDialog" title="新建文件夹" width="400px">
      <el-form :model="newFolderForm" :rules="newFolderRules" ref="newFolderFormRef">
        <el-form-item label="文件夹名称" prop="name">
          <el-input v-model="newFolderForm.name" placeholder="请输入文件夹名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-space>
          <el-button @click="showNewFolderDialog = false">取消</el-button>
          <el-button type="primary" @click="submitNewFolder">确定</el-button>
        </el-space>
      </template>
    </el-dialog>

    <el-dialog v-model="showRenameDialog" title="重命名" width="400px">
      <el-form :model="renameForm" :rules="renameRules" ref="renameFormRef">
        <el-form-item label="新名称" prop="newName">
          <el-input v-model="renameForm.newName" placeholder="请输入新名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-space>
          <el-button @click="showRenameDialog = false">取消</el-button>
          <el-button type="primary" @click="submitRename">确定</el-button>
        </el-space>
      </template>
    </el-dialog>

    <el-dialog v-model="showPoolStatsDialog" title="连接池状态" width="500px">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="活跃连接数">{{ poolStats.activeConnections }}</el-descriptions-item>
        <el-descriptions-item label="空闲连接数">{{ poolStats.idleConnections }}</el-descriptions-item>
        <el-descriptions-item label="最大连接数/每个key">{{ poolStats.maxTotalPerKey }}</el-descriptions-item>
        <el-descriptions-item label="最大空闲数/每个key">{{ poolStats.maxIdlePerKey }}</el-descriptions-item>
        <el-descriptions-item label="最小空闲数/每个key">{{ poolStats.minIdlePerKey }}</el-descriptions-item>
      </el-descriptions>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import { ftpApi } from '@/api/ftp'

const router = useRouter()
const uploadRef = ref(null)
const newFolderFormRef = ref(null)
const renameFormRef = ref(null)

const connected = ref(false)
const loading = ref(false)
const uploading = ref(false)
const showUploadDialog = ref(false)
const showNewFolderDialog = ref(false)
const showRenameDialog = ref(false)
const showPoolStatsDialog = ref(false)
const overwriteUpload = ref(false)
const uploadMode = ref('files')
const folderInputRef = ref(null)
const folderFileList = ref([])

const connectionConfig = ref(null)
const currentPath = ref('/')
const fileList = ref([])
const pathParts = ref(['/'])
const uploadFileList = ref([])
const poolStats = ref({})

const totalFiles = computed(() => fileList.value.length)
const directoryCount = computed(() => fileList.value.filter(f => f.isDirectory).length)
const fileCount = computed(() => fileList.value.filter(f => f.isFile).length)

const newFolderForm = reactive({ name: '' })
const newFolderRules = {
  name: [{ required: true, message: '请输入文件夹名称', trigger: 'blur' }]
}

const renameForm = reactive({ newName: '', oldPath: '', isDirectory: false })
const renameRules = {
  newName: [{ required: true, message: '请输入新名称', trigger: 'blur' }]
}

const goToSettings = () => {
  router.push('/settings')
}

const loadConnectionConfig = () => {
  const saved = sessionStorage.getItem('ftp_current_connection')
  if (saved) {
    try {
      connectionConfig.value = JSON.parse(saved)
      connected.value = true
    } catch (e) {
      connected.value = false
    }
  } else {
    const lastSaved = localStorage.getItem('ftp_last_connection')
    if (lastSaved) {
      try {
        connectionConfig.value = JSON.parse(lastSaved)
        connected.value = true
        sessionStorage.setItem('ftp_current_connection', lastSaved)
      } catch (e) {
        connected.value = false
      }
    }
  }
}

const refresh = async () => {
  if (!connected.value) return
  await listFiles(currentPath.value)
}

const listFiles = async (path) => {
  if (!connected.value) return
  
  loading.value = true
  try {
    const params = {
      ...connectionConfig.value,
      path
    }
    
    const result = await ftpApi.listFiles(params)
    fileList.value = result.data.files || []
    currentPath.value = result.data.currentPath || '/'
    pathParts.value = result.data.pathParts || ['/']
    
  } catch (error) {
    console.error('Failed to list files:', error)
  } finally {
    loading.value = false
  }
}

const navigateTo = (index) => {
  if (index === 0) {
    listFiles('/')
  } else {
    const path = '/' + pathParts.value.slice(1, index + 1).join('/')
    listFiles(path)
  }
}

const enterDirectory = (row) => {
  if (row.isDirectory) {
    const newPath = normalizePath(currentPath.value + '/' + row.name)
    listFiles(newPath)
  }
}

const handleRowDblclick = (row) => {
  if (row.isDirectory) {
    enterDirectory(row)
  } else {
    downloadFile(row)
  }
}

const handleNameClick = (row) => {
  if (row.isDirectory) {
    enterDirectory(row)
  }
}

const handleUploadChange = (file, fileList) => {
  uploadFileList.value = fileList
}

const handleUploadRemove = (file, fileList) => {
  uploadFileList.value = fileList
}

const triggerFolderSelect = () => {
  folderInputRef.value.click()
}

const handleFolderSelect = (event) => {
  const files = event.target.files
  if (!files || files.length === 0) {
    return
  }
  
  folderFileList.value = []
  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    const webkitRelativePath = file.webkitRelativePath || file.relativePath || file.name
    folderFileList.value.push({
      raw: file,
      name: file.name,
      relativePath: webkitRelativePath
    })
  }
  
  ElMessage.success(`已选择 ${folderFileList.value.length} 个文件`)
}

const closeUploadDialog = () => {
  showUploadDialog.value = false
  uploadFileList.value = []
  folderFileList.value = []
  if (folderInputRef.value) {
    folderInputRef.value.value = ''
  }
}

const startUpload = async () => {
  if (uploadMode.value === 'files' && uploadFileList.value.length === 0) {
    ElMessage.warning('请选择要上传的文件')
    return
  }
  if (uploadMode.value === 'folder' && folderFileList.value.length === 0) {
    ElMessage.warning('请选择要上传的文件夹')
    return
  }

  uploading.value = true
  const loadingInstance = ElLoading.service({
    lock: true,
    text: uploadMode.value === 'files' ? '正在上传文件...' : '正在上传文件夹...',
    background: 'rgba(0, 0, 0, 0.7)'
  })

  try {
    if (uploadMode.value === 'files') {
      for (const file of uploadFileList.value) {
        const formData = new FormData()
        formData.append('file', file.raw)
        
        const params = {
          ...connectionConfig.value,
          targetPath: normalizePath(currentPath.value + '/' + file.name),
          overwrite: overwriteUpload.value
        }
        
        for (const [key, value] of Object.entries(params)) {
          formData.append(key, value)
        }
        
        await ftpApi.uploadFile(formData)
        ElMessage.success(`文件 ${file.name} 上传成功`)
      }
    } else {
      const formData = new FormData()
      
      for (let i = 0; i < folderFileList.value.length; i++) {
        const fileItem = folderFileList.value[i]
        formData.append('files', fileItem.raw)
        formData.append('relativePaths', fileItem.relativePath)
      }
      
      const params = {
        ...connectionConfig.value,
        targetBasePath: currentPath.value,
        overwrite: overwriteUpload.value
      }
      
      for (const [key, value] of Object.entries(params)) {
        formData.append(key, value)
      }
      
      const result = await ftpApi.uploadFolder(formData)
      
      if (result.data) {
        if (result.data.isAllSuccess) {
          ElMessage.success(`文件夹上传成功，共 ${result.data.uploadedFiles} 个文件`)
        } else if (result.data.uploadedFiles > 0) {
          ElMessage.warning(`部分上传成功：${result.data.uploadedFiles}/${result.data.totalFiles} 个文件`)
        } else {
          ElMessage.error('上传失败')
        }
      }
    }
    
    closeUploadDialog()
    refresh()
    
  } catch (error) {
    console.error('Upload failed:', error)
  } finally {
    uploading.value = false
    loadingInstance.close()
  }
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const downloadFile = async (row) => {
  if (row.isDirectory) {
    ElMessage.warning('不能下载文件夹')
    return
  }

  const loadingInstance = ElLoading.service({
    lock: true,
    text: '正在下载文件...',
    background: 'rgba(0, 0, 0, 0.7)'
  })

  try {
    const filePath = normalizePath(currentPath.value + '/' + row.name)
    const params = {
      ...connectionConfig.value,
      path: filePath
    }
    
    const response = await fetch(`/api/ftp/download?${new URLSearchParams(params)}`)
    
    if (!response.ok) {
      throw new Error('Download failed')
    }
    
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = row.name
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('下载成功')
    
  } catch (error) {
    console.error('Download failed:', error)
    ElMessage.error('下载失败')
  } finally {
    loadingInstance.close()
  }
}

const createNewFolder = () => {
  newFolderForm.name = ''
  showNewFolderDialog.value = true
}

const submitNewFolder = async () => {
  await newFolderFormRef.value.validate()
  
  try {
    const newPath = normalizePath(currentPath.value + '/' + newFolderForm.name)
    const params = {
      ...connectionConfig.value,
      path: newPath
    }
    
    await ftpApi.createDirectory(params)
    ElMessage.success('文件夹创建成功')
    showNewFolderDialog.value = false
    refresh()
    
  } catch (error) {
    console.error('Create folder failed:', error)
  }
}

const openRenameDialog = (row) => {
  renameForm.oldPath = normalizePath(currentPath.value + '/' + row.name)
  renameForm.newName = row.name
  renameForm.isDirectory = row.isDirectory
  showRenameDialog.value = true
}

const submitRename = async () => {
  await renameFormRef.value.validate()
  
  try {
    const newPath = normalizePath(currentPath.value + '/' + renameForm.newName)
    const params = {
      ...connectionConfig.value,
      oldPath: renameForm.oldPath,
      newPath: newPath
    }
    
    await ftpApi.renameFile(params)
    ElMessage.success('重命名成功')
    showRenameDialog.value = false
    refresh()
    
  } catch (error) {
    console.error('Rename failed:', error)
  }
}

const handleDelete = async (row) => {
  const type = row.isDirectory ? '文件夹' : '文件'
  await ElMessageBox.confirm(
    `确定要删除${type} "${row.name}" 吗？${row.isDirectory ? '文件夹中的所有内容也将被删除。' : ''}`,
    '确认删除',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    }
  )

  try {
    const filePath = normalizePath(currentPath.value + '/' + row.name)
    const params = {
      ...connectionConfig.value,
      path: filePath
    }
    
    await ftpApi.deleteFile(params)
    ElMessage.success('删除成功')
    refresh()
    
  } catch (error) {
    console.error('Delete failed:', error)
  }
}

const normalizePath = (path) => {
  let normalized = path.replace(/\\/g, '/')
  while (normalized.includes('//')) {
    normalized = normalized.replace('//', '/')
  }
  if (!normalized.startsWith('/')) {
    normalized = '/' + normalized
  }
  return normalized
}

const isImage = (name) => {
  const ext = name.toLowerCase().split('.').pop()
  return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'].includes(ext)
}

const isZip = (name) => {
  const ext = name.toLowerCase().split('.').pop()
  return ['zip', 'rar', '7z', 'tar', 'gz', 'bz2'].includes(ext)
}

const isDocument = (name) => {
  const ext = name.toLowerCase().split('.').pop()
  return ['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'pdf', 'txt', 'md'].includes(ext)
}

onMounted(() => {
  loadConnectionConfig()
  if (connected.value) {
    listFiles('/')
  }
})
</script>

<style scoped>
.ftp-browser-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 10px 0;
}

.browser-card {
  margin-bottom: 30px;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
}

.browser-card :deep(.el-card__body) {
  padding: 25px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 16px;
  font-weight: 600;
  padding: 15px 20px;
}

.card-header :deep(.el-card__header) {
  padding: 15px 20px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.path-navigator {
  margin-bottom: 25px;
  padding: 12px 18px;
  background: #f5f7fa;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.path-btn {
  font-size: 14px;
  padding: 0;
}

.path-btn:hover {
  text-decoration: underline;
}

.file-name-cell {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.dir-icon {
  color: #409eff;
}

.file-icon {
  color: #909399;
}

.name-text {
  word-break: break-all;
}

.dir-size {
  color: #c0c4cc;
}

.not-connected {
  padding: 60px 0;
}

.pagination-info {
  margin-top: 20px;
  padding-top: 15px;
  border-top: 1px solid #ebeef5;
  text-align: right;
  color: #909399;
  font-size: 13px;
}

.folder-upload-area {
  border: 2px dashed #d9d9d9;
  border-radius: 4px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.3s;
}

.folder-upload-area:hover {
  border-color: #409eff;
}

.folder-upload-text {
  margin-top: 10px;
  font-size: 14px;
  color: #606266;
}

.folder-preview {
  margin-top: 20px;
}

.folder-file-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}

.folder-file-item {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid #ebeef5;
  gap: 8px;
}

.folder-file-item:last-child {
  border-bottom: none;
}

.folder-file-name {
  flex: 1;
  font-size: 13px;
  color: #606266;
  word-break: break-all;
}

.folder-file-size {
  font-size: 12px;
  color: #909399;
  min-width: 60px;
  text-align: right;
}

.more-files {
  padding: 8px 12px;
  text-align: center;
  font-size: 13px;
  color: #909399;
  background: #f5f7fa;
}
</style>
