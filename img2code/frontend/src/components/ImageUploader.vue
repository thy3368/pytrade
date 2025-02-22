&lt;template>
  &lt;div class="max-w-4xl mx-auto p-4">
    &lt;div class="mb-8">
      &lt;h1 class="text-3xl font-bold text-center mb-4">Screenshot to Vue Code Converter&lt;/h1>
      &lt;p class="text-center text-gray-600">Upload a screenshot and get Vue code with Tailwind CSS styling&lt;/p>
    &lt;/div>

    &lt;div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      &lt;!-- 左侧：图片上传区域 -->
      &lt;div class="border-2 border-dashed border-gray-300 rounded-lg p-6">
        &lt;div
          class="h-64 flex flex-col items-center justify-center"
          @dragover.prevent
          @drop.prevent="handleDrop"
          @click="triggerFileInput"
        >
          &lt;input
            type="file"
            ref="fileInput"
            class="hidden"
            accept="image/*"
            @change="handleFileSelect"
          >
          
          &lt;template v-if="!imagePreview">
            &lt;svg class="w-12 h-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              &lt;path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
            &lt;/svg>
            &lt;p class="text-gray-500">拖放图片到这里或点击上传&lt;/p>
          &lt;/template>
          
          &lt;img
            v-else
            :src="imagePreview"
            class="max-h-full object-contain"
            alt="Preview"
          >
        &lt;/div>
      &lt;/div>

      &lt;!-- 右侧：代码显示区域 -->
      &lt;div class="relative">
        &lt;div class="absolute top-0 right-0 z-10">
          &lt;button
            v-if="generatedCode"
            @click="copyCode"
            class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
          >
            {{ copied ? '已复制!' : '复制代码' }}
          &lt;/button>
        &lt;/div>
        
        &lt;div class="h-64 overflow-auto bg-gray-800 rounded-lg p-4">
          &lt;pre v-if="generatedCode" class="text-white">
            &lt;code>{{ generatedCode }}&lt;/code>
          &lt;/pre>
          &lt;div v-else class="h-full flex items-center justify-center text-gray-400">
            生成的代码将显示在这里
          &lt;/div>
        &lt;/div>
      &lt;/div>
    &lt;/div>

    &lt;div v-if="error" class="mt-4 p-4 bg-red-100 text-red-700 rounded">
      {{ error }}
    &lt;/div>

    &lt;div v-if="loading" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      &lt;div class="bg-white p-6 rounded-lg">
        &lt;div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500">&lt;/div>
        &lt;p class="mt-4 text-center">正在生成代码...&lt;/p>
      &lt;/div>
    &lt;/div>
  &lt;/div>
&lt;/template>

&lt;script setup>
import { ref } from 'vue'

const fileInput = ref(null)
const imagePreview = ref('')
const generatedCode = ref('')
const error = ref('')
const loading = ref(false)
const copied = ref(false)

const triggerFileInput = () => {
  fileInput.value.click()
}

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    processImage(file)
  }
}

const handleDrop = (event) => {
  const file = event.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    processImage(file)
  }
}

const processImage = (file) => {
  // 创建预览
  const reader = new FileReader()
  reader.onload = (e) => {
    imagePreview.value = e.target.result
  }
  reader.readAsDataURL(file)

  // 上传并生成代码
  generateCode(file)
}

const generateCode = async (file) => {
  loading.value = true
  error.value = ''
  generatedCode.value = ''

  try {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch('http://localhost:8000/api/convert', {
      method: 'POST',
      body: formData
    })

    const data = await response.json()
    
    if (data.error) {
      throw new Error(data.error)
    }

    generatedCode.value = data.code
  } catch (err) {
    error.value = `生成代码时出错: ${err.message}`
  } finally {
    loading.value = false
  }
}

const copyCode = () => {
  if (generatedCode.value) {
    navigator.clipboard.writeText(generatedCode.value)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  }
}
&lt;/script>

&lt;style scoped>
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
}
&lt;/style>
