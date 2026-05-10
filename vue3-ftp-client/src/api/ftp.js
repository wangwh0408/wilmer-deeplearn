import axios from 'axios'
import { ElMessage } from 'element-plus'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.request.use(
  config => {
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

api.interceptors.response.use(
  response => {
    const res = response.data
    if (res.code === 200 || res.code === 0) {
      return res
    }
    ElMessage.error(res.message || '请求失败')
    return Promise.reject(new Error(res.message || '请求失败'))
  },
  error => {
    let message = '网络错误'
    if (error.response) {
      switch (error.response.status) {
        case 400:
          message = '请求参数错误'
          break
        case 403:
          message = '权限不足'
          break
        case 404:
          message = '资源不存在'
          break
        case 408:
          message = '请求超时'
          break
        case 500:
          message = '服务器内部错误'
          break
        case 503:
          message = '服务不可用'
          break
        default:
          message = `请求失败 (${error.response.status})`
      }
    } else if (error.code === 'ECONNABORTED') {
      message = '请求超时，请重试'
    } else if (error.message.includes('timeout')) {
      message = '请求超时，请重试'
    }
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

export const ftpApi = {
  testConnection(params) {
    return api.post('/ftp/test', params)
  },

  listFiles(params) {
    return api.get('/ftp/list', { params })
  },

  getFileInfo(params) {
    return api.get('/ftp/file-info', { params })
  },

  uploadFile(formData, onUploadProgress) {
    return api.post('/ftp/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress
    })
  },

  uploadFolder(formData, onUploadProgress) {
    return api.post('/ftp/upload-folder', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress
    })
  },

  downloadFile(params) {
    return api.get('/ftp/download', {
      params,
      responseType: 'blob'
    })
  },

  createDirectory(params) {
    return api.post('/ftp/create-dir', null, { params })
  },

  deleteFile(params) {
    return api.delete('/ftp/delete', { params })
  },

  renameFile(params) {
    return api.post('/ftp/rename', null, { params })
  },

  copyFile(params) {
    return api.post('/ftp/copy', null, { params })
  },

  getPoolStats() {
    return api.get('/ftp/pool-stats')
  },

  getApiInfo() {
    return api.get('/ftp/info')
  }
}

export default api
