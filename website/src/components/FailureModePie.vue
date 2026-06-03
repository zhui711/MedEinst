<script setup>
import { failureModes } from '@/data/failureModes'
import PieChart from './PieChart.vue'
</script>

<template>
  <div class="pie-grid">
    <PieChart v-for="model in failureModes" :key="model.model" :model="model" />
  </div>
</template>
