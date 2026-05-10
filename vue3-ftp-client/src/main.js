import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import App from './App.vue'
import FtpBrowser from './views/FtpBrowser.vue'
import FtpSettings from './views/FtpSettings.vue'

const routes = [
  {
    path: '/',
    redirect: '/ftp'
  },
  {
    path: '/ftp',
    name: 'FtpBrowser',
    component: FtpBrowser,
    meta: { title: 'FTP文件浏览器' }
  },
  {
    path: '/settings',
    name: 'FtpSettings',
    component: FtpSettings,
    meta: { title: '设置' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'FTP Client'
  next()
})

const app = createApp(App)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.use(router)
app.use(ElementPlus, { locale: zhCn })

app.mount('#app')
