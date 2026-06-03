<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { PieChart as EChartsPieChart } from 'echarts/charts'
import { TooltipComponent } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'

echarts.use([EChartsPieChart, TooltipComponent, SVGRenderer])

const props = defineProps({
  model: {
    type: Object,
    required: true,
  },
})

const chartEl = ref(null)
let chart = null

const chartLabels = {
  Blindness: 'Blind',
  Underthinking: 'Under',
  Overthinking: 'Over',
}

function optionForModel() {
  return {
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const data = params.data
        return [
          `<strong>${props.model.model}</strong>`,
          `${data.category}: ${data.value} (${data.percentage}%)`,
        ].join('<br>')
      },
    },
    series: [
      {
        name: 'Attribution',
        type: 'pie',
        radius: ['34%', '58%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2,
        },
        label: {
          show: true,
          position: 'outside',
          formatter: (params) => params.data.name,
          fontSize: 12,
          fontWeight: 700,
          color: '#24324b',
        },
        labelLine: {
          show: true,
          length: 10,
          length2: 6,
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 13,
            fontWeight: 700,
          },
        },
        data: props.model.categories.map((category) => ({
          ...category,
          name: chartLabels[category.category] ?? category.category,
          itemStyle: { color: category.color },
        })),
      },
    ],
  }
}

function resizeChart() {
  chart?.resize()
}

function renderChart() {
  if (!chartEl.value) return

  if (!chart) {
    chart = echarts.init(chartEl.value, null, { renderer: 'svg' })
    window.addEventListener('resize', resizeChart)
  }

  chart.setOption(optionForModel(), true)
  resizeChart()
}

onMounted(async () => {
  await nextTick()
  renderChart()
})

watch(() => props.model, renderChart, { deep: true })

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeChart)
  chart?.dispose()
  chart = null
})
</script>

<template>
  <article class="pie-card">
    <div class="pie-card-heading">
      <h3>{{ model.model }}</h3>
      <span>(N={{ model.total }} Bias Cases)</span>
    </div>
    <div ref="chartEl" class="pie-chart" role="img" :aria-label="`${model.model} failure mode attribution pie chart`" />
  </article>
</template>
