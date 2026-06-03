<script setup>
import { performanceGroups } from '@/data/performance'

function isHighlighted(row) {
  return row[0] === 'ECR-Agent (Qwen3-32B)'
}
</script>

<template>
  <div class="table-copy">
    <p>Baseline Acc and Robust Acc are higher-is-better. Bias Trap Rate is lower-is-better.</p>
  </div>

  <div class="table-scroll" tabindex="0" aria-label="Overall Performance on MedEinst table">
    <table class="performance-table">
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col">Size</th>
          <th scope="col">Baseline Acc ↑</th>
          <th scope="col">Robust Acc ↑</th>
          <th scope="col">Bias Trap Rate ↓</th>
        </tr>
      </thead>
      <tbody>
        <template v-for="group in performanceGroups" :key="group.name">
          <tr class="group-row">
            <th colspan="5" scope="colgroup">{{ group.name }}</th>
          </tr>
          <tr
            v-for="row in group.rows"
            :key="row[0]"
            :class="{ 'highlight-row': isHighlighted(row) }"
          >
            <th scope="row">{{ row[0] }}</th>
            <td>{{ row[1] }}</td>
            <td>{{ row[2] }}</td>
            <td>{{ row[3] }}</td>
            <td>{{ row[4] }}</td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</template>
