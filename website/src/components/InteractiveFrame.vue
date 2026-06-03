<script setup>
defineProps({
  src: {
    type: String,
    required: true,
  },
  title: {
    type: String,
    required: true,
  },
  caption: {
    type: String,
    default: '',
  },
  height: {
    type: String,
    default: '720px',
  },
})
</script>

<template>
  <figure class="interactive-frame-panel">
    <iframe
      :src="src"
      :title="title"
      :style="{ height }"
      loading="lazy"
    />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>
