<template>
  <div class="app-container">
    <header class="header">
      <h1>智能知识图谱关系挖掘分析系统</h1>
    </header>

    <main class="main-content">
      <!-- 输入区域 -->
      <section class="input-section">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <span>语义分析输入</span>
            </div>
          </template>

          <el-form
              ref="formRef"
              :model="form"
              :rules="rules"
              @submit.prevent="handleSubmit"
          >
            <el-form-item label="句子内容" prop="sentence">
              <el-input
                  v-model="form.sentence"
                  type="textarea"
                  :rows="3"
                  placeholder="请输入完整句子"
              />
            </el-form-item>

            <div class="entity-row">
              <el-form-item label="实体1" prop="entity1">
                <el-input
                    v-model="form.entity1"
                    placeholder="请输入第一个实体"
                />
              </el-form-item>

              <el-form-item label="实体2" prop="entity2">
                <el-input
                    v-model="form.entity2"
                    placeholder="请输入第二个实体"
                />
              </el-form-item>
            </div>

            <el-button
                type="primary"
                native-type="submit"
                :loading="isLoading"
                class="submit-btn"
            >
              开始分析
            </el-button>
          </el-form>
        </el-card>
      </section>

      <!-- 结果展示 -->
      <section v-if="result" class="result-section">
        <el-row :gutter="20">
          <!-- 关系卡片 -->
          <el-col :span="12">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <el-icon>
                    <connection/>
                  </el-icon>
                  <span>实体关系</span>
                </div>
              </template>

              <div class="relation-info">
                <el-tag :type="confidenceTagType" effect="dark">
                  {{ result.relation }}
                </el-tag>
                <div class="confidence-meter">
                  <el-progress
                      :percentage="result.confidence * 100"
                      :color="confidenceColor"
                      :stroke-width="16"
                      :format="formatConfidence"
                  />
                </div>
              </div>
            </el-card>
          </el-col>

          <!-- 可视化图表 -->
          <el-col :span="12">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <el-icon>
                    <data-analysis/>
                  </el-icon>
                  <span>置信度趋势</span>
                </div>
              </template>

              <div ref="chartRef" class="chart-container"></div>
            </el-card>
          </el-col>
        </el-row>
      </section>

      <!-- 历史记录 -->
      <section class="history-section">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon>
                <clock/>
              </el-icon>
              <span>分析历史</span>
            </div>
          </template>

          <el-table
              :data="history"
              stripe
              style="width: 100%"
              v-loading="historyLoading"
          >
            <el-table-column prop="sentence" label="句子" min-width="200"/>
            <el-table-column prop="entity1" label="实体1" width="120"/>
            <el-table-column prop="entity2" label="实体2" width="120"/>
            <el-table-column prop="relation" label="关系" width="150">
              <template #default="{ row }">
                <el-tag>{{ row.relation }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="confidence" label="置信度" width="120">
              <template #default="{ row }">
                {{ (row.confidence * 100).toFixed(1) }}%
              </template>
            </el-table-column>
            <el-table-column prop="create_time" label="时间" width="180"/>
          </el-table>
        </el-card>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import {ref, reactive, onMounted, nextTick, computed} from 'vue'
import {ElMessage, type FormInstance, type FormRules} from 'element-plus'
import {Connection, DataAnalysis, Clock} from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import axios from 'axios'

interface PredictionResult {
  relation: string
  confidence: number
}

interface HistoryRecord {
  sentence: string
  entity1: string
  entity2: string
  relation: string
  confidence: number
  create_time: string
}

// 表单数据
const form = reactive({
  sentence: '',
  entity1: '',
  entity2: ''
})

// 表单验证规则
const rules = reactive<FormRules>({
  sentence: [
    {required: true, message: '请输入句子内容', trigger: 'blur'},
    {min: 10, message: '句子长度至少10个字符', trigger: 'blur'}
  ],
  entity1: [
    {required: true, message: '请输入实体1', trigger: 'blur'}
  ],
  entity2: [
    {required: true, message: '请输入实体2', trigger: 'blur'}
  ]
})

// 图表实例
const chartRef = ref<HTMLElement>()
let chart: echarts.ECharts | null = null

// 状态管理
const formRef = ref<FormInstance>()
const isLoading = ref(false)
const result = ref<PredictionResult | null>(null)
const history = ref<HistoryRecord[]>([])
const historyLoading = ref(false)

// 置信度颜色计算
const confidenceColor = computed(() => {
  if (!result.value) return '#909399'
  const confidence = result.value.confidence
  return confidence > 0.8 ? '#67c23a' : confidence > 0.5 ? '#e6a23c' : '#f56c6c'
})

// 置信度标签类型
const confidenceTagType = computed(() => {
  if (!result.value) return 'info'
  const confidence = result.value.confidence
  return confidence > 0.8 ? 'success' : confidence > 0.5 ? 'warning' : 'danger'
})

// 初始化图表
const initChart = () => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value)
    const option = {
      xAxis: {type: 'category', data: []},
      yAxis: {type: 'value', max: 1},
      series: [{type: 'line', smooth: true, data: []}]
    }
    chart.setOption(option)
  }
}
// 更新图表数据
// const updateChart = (confidence: number) => {
//   if (!chart) return
//
//   const option = chart.getOption()
//   const xData = option.xAxis[0].data as string[]
//   const yData = option.series[0].data as number[]
//
//   xData.push(new Date().toLocaleTimeString())
//   yData.push(confidence)
//
//   if (xData.length > 10) {
//     xData.shift()
//     yData.shift()
//   }
//
//   chart.setOption({
//     xAxis: {data: xData},
//     series: [{data: yData}]
//   })
// }

// 提交表单
const handleSubmit = async () => {
  try {
    await formRef.value?.validate()
    isLoading.value = true
    const rawData = `{"sentence":"${form.sentence}","entity1":"${form.entity1}","entity2":"${form.entity2}"}`

    const response = await axios.post('http://10.70.233.54:8090/api/predict', rawData,
        {
          timeout: 15000,  // 增加超时时间
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          },
        })

    result.value = response.data
    // updateChart(response.data.confidence)
    // await fetchHistory()
    ElMessage.success('分析成功')
  } catch (error) {
    // 增强错误处理
    if (axios.isAxiosError(error)) {
      if (!error.response) {
        ElMessage.error('无法连接到服务器')
      } else if (error.response.status === 504) {
        ElMessage.error('请求超时，请检查网络连接')
      } else {
        ElMessage.error(`服务器错误: ${error.response.data.message}`)
      }
    }
  } finally {
    isLoading.value = false
  }
}

// 获取历史记录
// const fetchHistory = async () => {
//   try {
//     historyLoading.value = true
//     const response = await axios.get('/api/history')
//     history.value = response.data
//   } catch (error) {
//     ElMessage.error('获取历史记录失败')
//   } finally {
//     historyLoading.value = false
//   }
// }

// 格式化置信度显示
const formatConfidence = (percentage: number) => {
  return `${percentage.toFixed(1)}% 置信度`
}

onMounted(() => {
  nextTick(initChart)
  // fetchHistory()
})
</script>

<style lang="scss" scoped>
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;

  .header {
    text-align: center;
    margin-bottom: 30px;

    h1 {
      color: var(--el-color-primary);
      font-size: 2.2rem;
    }
  }

  .main-content {
    display: grid;
    gap: 20px;
  }

  .entity-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .submit-btn {
    width: 100%;
    margin-top: 15px;
  }

  .result-section {
    margin-top: 20px;

    .relation-info {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;

      .el-tag {
        font-size: 1.2rem;
        padding: 12px 20px;
      }
    }

    .chart-container {
      height: 300px;
    }
  }

  .history-section {
    margin-top: 20px;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: bold;
  }
}
</style>