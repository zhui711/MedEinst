<script setup>
defineProps({
  src: {
    type: String,
    required: true,
  },
  alt: {
    type: String,
    required: true,
  },
  caption: {
    type: String,
    required: true,
  },
})
</script>

<template>
  <figure class="figure-panel">
    <img :src="src" :alt="alt" loading="lazy" />
    <figcaption>{{ caption }}</figcaption>
  </figure>
</template>
