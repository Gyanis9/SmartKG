import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import App from './App.vue'
import 'element-plus/dist/index.css'

const app = createApp(App)

// 配置axios
import axios from 'axios'
// axios.defaults.baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8090'

app.use(ElementPlus)
app.mount('#app')