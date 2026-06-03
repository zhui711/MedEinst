<script setup>
defineProps({
  stats: {
    type: Array,
    required: true,
  },
})
</script>

<template>
  <div class="stat-grid" aria-label="Benchmark statistics">
    <article v-for="stat in stats" :key="stat.label" class="stat-card">
      <strong>{{ stat.value }}</strong>
      <span>{{ stat.label }}</span>
    </article>
  </div>
</template>
